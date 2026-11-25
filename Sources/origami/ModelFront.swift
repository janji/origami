import CryptoKit
import Darwin
import Foundation
import OrigamiEngine

/// `origami serve`: the front clients talk to, as Ollama's server is. It lists the configured model and every other
/// one `origami model list` holds (the list is read at each request, so a model added while it runs shows up), and
/// runs one of them at a time in a child process, the one-model server (`origami serve --single-model`), passing each
/// request to it and its response back unchanged. A request naming another listed model waits for the running
/// requests (`ModelScheduler`), stops that child, whose memory all comes back when it exits, and starts the named
/// one. A name not listed goes to the running model, as one server answering under any name always did (wrappers
/// such as llama-swap rename models). The front answers the listings itself (`/v1/models`, `/api/tags`,
/// `/api/version`, `/api/show` for a model not running), the dashboard page, the dashboard's jobs (`/v1/jobs`), model
/// search (`/v1/models/found`, `/v1/models/check`), folder browser (`/v1/fs`, `Folders`) and storage map (`/v1/storage`, `Storage`) and, while no
/// child runs, `/health`, `/v1/status` and `/api/ps`. It
/// stays up when its model cannot start, saying why, so the dashboard can fix it.
final class ModelFront: @unchecked Sendable {
    /// A listed model: its id, the `--model` its child gets (nil: the front's own), what Ollama clients show, and the
    /// `model` setting that makes it the default (nil: none, the automatic choice), and what its availability is kept
    /// under: its file or folder, or `locate:<setting>` for a repo id or the automatic choice, found where downloads go.
    struct Row {
        let id: String, spec: String?, family: ModelFamily, bytes: UInt64, quantization: String, added: String, setting: String?,
            target: String
        /// The exact model a low-bit expert variant is based on (`model add <experts.gguf> --base`), nil for any other model.
        var variantOf: String? = nil
    }
    final class Child {
        let process: Process, port: UInt16
        let configuration: [String: String]
        init(_ process: Process, port: UInt16, configuration: [String: String]) {
            self.process = process
            self.port = port
            self.configuration = configuration
        }
    }

    /// As many as the model's server queues, so a request the front lets through is not refused there.
    static let maximumConnections = LocalHTTPConfiguration().maximumConnections
    private let flags: [String: [String]], log: @Sendable (String) -> Void
    /// `listen lan` as the front started (it binds once): other devices reach it, with `remoteKey`, and its children
    /// take this Mac's network names as a Host too. `port` is the one it listens on.
    let lan: Bool, port: Int
    private let lock = NSLock()
    private let dashboardSession = UUID().uuidString
    private let frontInstance = UUID().uuidString
    private let observationTags: [String: String]
    /// The settings and the configured model's id (`servedID`), read again by `POST v1/reload`.
    private var loaded: (settings: Settings, firstID: String)
    var settings: Settings { lock.withLock { loaded.settings } }
    var firstID: String { lock.withLock { loaded.firstID } }
    /// The model the front started with, listed even after a reload makes another the default, so a folder that was
    /// never registered stays one choice away.
    private let startup: (id: String, setting: String?)
    private(set) var scheduler: ModelScheduler<Child>!
    private var starting: String?, launching: Process?, failure: String?, stopped = false
    /// The starting model's output, for its progress (`loadingProgress`).
    private var startingLog: LastError?
    /// The loaded model's drive seen gone at this many checks in a row (`checkModelDrive`): from the second it is
    /// `degraded` until the drive is mounted again (the model process reopens its files by itself, `ModelDrive`).
    private var driveMisses = 0
    /// `slowed_by` (`markSlowed`): when each listed condition was first seen, and the last list with its time, which
    /// `/health` reuses for 5 s.
    private var slowSeen: [String: String] = [:]
    private var lastSlowed: (at: Date, list: [[String: Any]])?
    private let connectionBudget = HTTPConnectionBudget(maximumRequests: maximumConnections)
    /// Wrong keys from another device (or a trusted proxy's client): 10 a minute, then 429 until the minute ends.
    private let authLimiter = HTTPAuthLimiter()
    private let httpMetrics: HTTPRequestMetrics
    /// The accept loop and its request threads are metric producers. Keep registration under shutdown ownership.
    private let requests = DispatchGroup()
    private var clientDescriptors = Set<Int32>()
    /// Stop and unload (`POST v1/stop`, saved as `stopped`): no model runs until `POST v1/start`. `unloading` while the
    /// running requests get their wait and the child stops (`cancelling` once they are cancelled); `resumeID` is the
    /// model Start loads again, `held` what the last stopped child held and `freed` that for the stop, with `unloadNote`.
    private var unloaded: Bool, unloading = false, cancelling = false, resumeID: String?, held: UInt64?, freed: UInt64?, unloadNote: String?
    /// A reload waiting for the running reply, and why (`pending` in `/health`, `/v1/status` and the dashboard's state,
    /// which the model that still serves answers meanwhile); nil from the moment that model is being stopped.
    private var pendingReload: String?
    /// `idle_unload_minutes`: the last look that found the model stopped, loading, replying or a job running.
    private var activeAt = Date()
    /// `origami tune --memory`: the limit the next model loads at instead of the saved one, until a reload without it;
    /// `trialModel` the model it measures (the one loaded when it began, not necessarily the default); `trialEnded` once an
    /// app's request ended it (no further trial loads until its job ends, or a terminal search its restore);
    /// `reloadsUnderway` counts reloads not yet done.
    private var memoryTrial: UInt64?, trialModel: String?, trialEnded = false, reloadsUnderway = 0
    /// One app request at a time ends a trial, so a second waits for the first one's restore.
    private let trialEnd = NSLock()
    /// How long a stop waits for the running requests before it cancels them.
    static let unloadWait: TimeInterval = 30
    /// How long a model's process may take to listen (others listen only once loaded): as long as a request waits.
    static let startWait: TimeInterval = LocalHTTPConfiguration().queueTimeout
    static let stoppedFailure = ServingFailure(
        "Origami is stopped; press Start on its dashboard (this server's address in a browser) or run origami start on its Mac",
        status: 503, type: "server_unavailable", code: "stopped")
    /// The drive a model is on that macOS has not let Origami read (`drive`, `path`), while that keeps it from starting.
    private var permission: [String: String]?
    let jobs: Jobs
    /// `memory_auto_measure`: when the last request came, and the memory limit measured while idle.
    private let autoMeasure = AutoMeasure()
    /// The model menu's one source of truth, kept by the front whichever model runs: each listed model's last check
    /// (`Stores.Availability`, and a GGUF's size) with the checks still running, and the last complete model search with
    /// whether another runs. Both are learned off the request path, so a sleeping or unreadable drive never holds a
    /// listing, and what was last known answers while a check runs.
    private var checked: [String: (state: Stores.Availability, bytes: UInt64?, date: Date)] = [:], checking: [String: Date] = [:]
    /// Each checked model's recommended sampling (`Models.recommendedSampling`), read by its availability check.
    private var recommendations: [String: SamplingDefaults] = [:]
    private var foundCache: (date: Date, models: [[String: Any]], unreadable: [String], partial: String?, known: Set<String>)?,
        searching = false, searchAgain = false
    /// The registry as last read, listed if a read fails (a file being replaced by hand), so models never drop out.
    private var retiredModels = Set<String>()
    private var lastEntries: [[String: Any]] = []

    init(settings: Settings, flags: [String: [String]], port: Int, log: @escaping @Sendable (String) -> Void) {
        var tags =
            (try? Commands.observationTags(settings, version: version, identity: "not_selected", context: settings.context)) ?? [
                "engine_version": version, "provenance": "unavailable",
            ]
        tags["model_identity_scope"] = "front; selected worker identity is recorded by worker"
        observationTags = tags
        // Front and worker use separate files: their recorder locks are process-local.
        if let folder = settings.requestLog,
            let recorder = try? RequestRecorder(folder: folder.appendingPathComponent("front", isDirectory: true))
        {
            httpMetrics = HTTPRequestMetrics(scope: "front", record: { recorder.record($0, units: [:]) })
        } else {
            httpMetrics = HTTPRequestMetrics(scope: "front")
        }
        self.flags = flags
        self.log = log
        lan = settings.listenLAN
        self.port = port
        unloaded = settings.stopped
        Tuning.removeTrialHistories()  // a trial limit lives in this process only: none is left from before
        Tuning.settleAutoMeasures(settings)
        // Jobs read the same settings file and stores; the port, the model and a conversion's width are the job's own.
        jobs = Jobs(
            options: flags.sorted { $0.key < $1.key }.filter { !["port", "model", "expert-bits"].contains($0.key) }
                .flatMap { key, values in values.map { "--\(key)=\($0)" } }, log: log)
        loaded = (settings, Self.servedID(settings))
        startup = (loaded.firstID, settings.model)
        scheduler = ModelScheduler(
            start: { [unowned self] in try self.start($0) },
            ready: { [unowned self] id in
                do { try self.ready(id) } catch {
                    self.log("not switching to \(id): \(ServingFailure.from(error).description)")
                    throw error
                }
            }, stop: { [unowned self] in self.stop($0) },
            alive: { $0.process.isRunning },
            // Alternating models: up to as many of the loaded model's requests as are buffered at once go ahead of one
            // for another model, and none once it has waited a quarter of the queue's limit (15 minutes).
            batch: LocalHTTPConfiguration().bufferedRequests, batchSeconds: LocalHTTPConfiguration().queueTimeout / 4)
    }

    /// The one name a server lists without `list_models`: whatever the dashboard set up.
    static let setupID = "origami"
    /// The id a one-model server lists: `origami`, or with `list_models` the model's own (`servedID`).
    static func listedID(_ settings: Settings) -> String { settings.listModels ? servedID(settings) : setupID }

    /// The id a one-model server answers to: the model's registered name, else its family's id (`qwen3.8-flash-next`…).
    static func servedID(_ settings: Settings) -> String {
        guard let spec = settings.model else { return ModelFamily.qwen.rawValue }
        if ModelRegistry.entry(spec, settings: settings) != nil { return spec }
        let path = Settings.expand(spec)
        // A missing .gguf file is named after its file (or the entry registered for it, so the menu lists it once), never
        // taken for the default Qwen model.
        if path.lowercased().hasSuffix(".gguf") && !FileManager.default.fileExists(atPath: path) {
            let entry = (try? ModelRegistry.entries(settings))?.first { ($0["path"] as? String).map(Settings.expand) == path }
            return entry?["name"] as? String ?? URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
        }
        if Models.isGGUF(path),
            let family = (try? GGUFSource(URL(fileURLWithPath: path))).flatMap({ ModelFamily.detect(gguf: $0.metadata) })
        {
            return family.rawValue
        }
        return ModelFamily.qwen.rawValue
    }

    /// The configured model first, then every other registered one, then the model the front started with if it is
    /// neither.
    func rows() -> [Row] {
        let (settings, firstID) = lock.withLock { loaded }
        let entries: [[String: Any]]
        if let read = try? ModelRegistry.entries(settings) {
            entries = read
            lock.withLock { lastEntries = read }
        } else {
            entries = lock.withLock { lastEntries }
        }
        func row(_ entry: [String: Any]?, id: String, spec: String?, setting: String?) -> Row {
            let path = entry?["path"] as? String
            let chosen = spec ?? settings.model
            let target = path ?? chosen.map { "/~.".contains($0.first ?? " ") ? Settings.expand($0) : "locate:" + $0 } ?? "locate:"
            // A GGUF's size comes from its check (no file is touched here); a split one counts its first part only.
            let bytes =
                ((entry?["manifest"] as? [String: Any])?["bytes"] as? NSNumber)?.uint64Value ?? lock.withLock { checked[target]?.bytes }
                ?? 0
            let family = (entry?["family"] as? String).flatMap(ModelFamily.init(rawValue:)) ?? ModelFamily(rawValue: id) ?? .qwen
            return Row(
                id: id, spec: spec, family: family, bytes: bytes,
                quantization: "unknown",
                added: entry?["added"] as? String ?? "2026-01-01T00:00:00Z", setting: setting, target: target,
                variantOf: entry?["base"] as? String)
        }
        var rows =
            [row(entries.first { $0["name"] as? String == firstID }, id: firstID, spec: nil, setting: settings.model)]
            + entries.compactMap { entry in
                (entry["name"] as? String).flatMap { $0 == firstID ? nil : row(entry, id: $0, spec: $0, setting: $0) }
            }
        if !rows.contains(where: { $0.id == startup.id }) {
            rows.append(
                row(nil, id: startup.id, spec: startup.setting.map(Settings.expand) ?? Models.defaultRepo, setting: startup.setting))
        }
        return rows.filter { row in
            entries.contains { $0["name"] as? String == row.id } || !lock.withLock { retiredModels.contains(row.id) }
        }
    }

    /// `POST v1/reload`: the settings are read again and the default model restarts with them once the running
    /// requests finish (the dashboard's Apply); answered at once. Until the running reply ends, the model that still
    /// serves answers `/health` and status, with `pending` saying why it will reload; `loading` from its stop until back.
    private func reload(because why: String = "to apply the saved settings", trial: UInt64? = nil) throws -> String {
        guard !isUnloaded else { throw Self.stoppedFailure }
        let fresh: Settings
        do { fresh = try Settings.load(flags: flags) } catch { throw ServingFailure("The settings cannot be read: \(error)") }
        let why = trial.map { "to measure a memory limit of \(OrigamiPlan.gigabytes($0))" } ?? why
        let running = scheduler.current?.id
        let id = try lock.withLock { () throws -> String in
            if trial != nil, trialEnded {
                throw ServingFailure("A request arrived, so the memory search stops; your saved limit is back", status: 409)
            }
            // A memory search measures the model loaded when it began, and its end reloads that model.
            let id = trial != nil || memoryTrial != nil ? trialModel ?? running ?? Self.servedID(fresh) : Self.servedID(fresh)
            loaded = (fresh, Self.servedID(fresh))
            pendingReload = Self.pendingText + " " + why
            memoryTrial = trial
            trialModel = trial == nil ? nil : id
            reloadsUnderway += 1
            return id
        }
        log("reloading \(id) \(why), once the running requests end")
        Thread.detachNewThread { [self] in
            do { try scheduler.reload(id) } catch { log("reload: \(ServingFailure.from(error).description)") }
            lock.withLock {
                pendingReload = nil
                reloadsUnderway -= 1
            }
        }
        return id
    }
    static let pendingText = "reload after the current reply"
    /// The 202 answer to a reload: `pending` when a reply runs, so the page says the model reloads after it.
    private func reloading(_ id: String, _ more: [String: Any] = [:]) -> [String: Any] {
        more.merging(["reloading": true, "model": id, "pending": scheduler.busy ? Self.pendingText : NSNull()]) { $1 }
    }

    /// After a Qwen hot-copy job: the model reloads only when the job changed what it reads (a copy built, resized, moved
    /// or removed: `Stores.signature`) or the saved change alone does (the copy turned off or on, or its folder). The
    /// job's closing line.
    private func afterHotCopy(_ state: String, before: [String: UInt64]) -> String? {
        guard state != "cancelled" else { return nil }
        let fresh = (try? Settings.load(flags: flags)) ?? settings
        let after = Stores.signature(fresh)
        let built = after.filter { before[$0.key] == nil }.values.reduce(0, +)
        let now = DashboardState.configuration(fresh)
        let moved =
            scheduler.current.map {
                $0.child.configuration["hot_copy_dir"] != now["hot_copy_dir"]
                    || ($0.child.configuration["hot_copy"] == "off") != (now["hot_copy"] == "off")
            } ?? false
        guard after != before || moved, (try? reload(because: "to use the new hot copy")) != nil else {
            return state == "done" ? "Nothing changed, so the model keeps running as it is." : nil
        }
        return (built > 0 ? "\(OrigamiPlan.gigabytes(built)) copied; the model" : "The model")
            + " reloads after the current reply to use it."
    }

    var isStopped: Bool { lock.withLock { stopped } }
    var isUnloaded: Bool { lock.withLock { unloaded } }

    /// `POST v1/stop`: saved as `stopped` (a restart stays stopped), then answered at once while the running requests
    /// get `unloadWait` to finish, are cancelled if they have not, and the child stops, giving all its memory back.
    /// Meanwhile and after, requests get `stoppedFailure`, and `/health` says `stopping`, then `stopped`.
    private func unload() throws -> [String: Any] {
        guard !isUnloaded else { return stoppedState() }
        try Settings.write(["stopped": true], to: settings.configFile)
        let running = scheduler.current?.id
        // Two stops at once: the first one unloads.
        guard
            lock.withLock({ () -> Bool in
                guard !unloaded else { return false }
                unloaded = true
                unloading = true
                resumeID = running ?? resumeID
                held = nil
                freed = nil
                unloadNote = nil
                memoryTrial = nil  // a stopped `tune --memory` cannot restore it; Start loads the saved limit
                trialModel = nil
                launching?.terminate()  // a model still starting
                return true
            })
        else { return stoppedState() }
        log("stopping and unloading \(running ?? "(no model running)")")
        Thread.detachNewThread { [self] in
            let cancelled: Int
            do { cancelled = try scheduler.unload(within: Self.unloadWait) { [self] in lock.withLock { cancelling = true } } } catch {
                cancelled = 0
            }
            let note =
                cancelled > 0
                ? "\(cancelled == 1 ? "A request" : "\(cancelled) requests") still running after \(Int(Self.unloadWait)) s \(cancelled == 1 ? "was" : "were") cancelled."
                : nil
            let given = lock.withLock { () -> UInt64? in
                cancelling = false
                unloading = false
                unloadNote = note
                freed = running == nil ? nil : held
                return freed
            }
            log(
                "stopped: no model loaded" + (given.map { ", about \(OrigamiPlan.gigabytes($0)) back to macOS" } ?? "")
                    + (note.map { "; " + $0 } ?? ""))
        }
        return stoppedState()
    }

    /// `POST v1/start`: `stopped` is cleared and the model that ran before the stop (else the default) starts again with
    /// the saved settings, read afresh; answered at once, `/health` says `loading` until it runs. After a failed start (a
    /// drive not readable yet), the same: a stop and a start in one (`retried`, with the failure).
    private func resume() throws -> (Int, [String: Any]) {
        let (was, busy, previous, failed) = lock.withLock { (unloaded, unloading, resumeID, starting == nil ? failure : nil) }
        guard !busy else { throw ServingFailure("Origami is still unloading the model; start it once it has stopped", status: 409) }
        let retry = !was && failed != nil && scheduler.current == nil
        guard was || retry else { return (200, ["status": "running", "model": scheduler.current?.id ?? firstID]) }
        if was { try Settings.write(["stopped": nil], to: settings.configFile) }
        let fresh = (try? Settings.load(flags: flags)) ?? settings
        lock.withLock {
            loaded = (fresh, Self.servedID(fresh))
            unloaded = false
            freed = nil
            unloadNote = nil
            failure = nil
        }
        let target = fresh.model != nil ? firstID : previous.flatMap { id in rows().contains { $0.id == id } ? id : nil } ?? firstID
        log("starting \(target) again" + (retry ? " after a failed start" : ""))
        Thread.detachNewThread { [self] in if (try? scheduler.acquire(target, fallback: firstID)) != nil { scheduler.release() } }
        var body: [String: Any] = ["status": "loading", "model": target]
        if retry { (body["retried"], body["failure"]) = (true, failed) }
        return (202, body)
    }

    /// `/health` and `/v1/status` while stopped, as a 503 with the error apps get: what the stop freed (the stopped
    /// child's footprint) and this Mac's memory, for the dashboard's memory strip.
    private func stoppedState() -> [String: Any] {
        lock.withLock {
            [
                "status": unloading ? "stopping" : "stopped", "version": version, "model": resumeID ?? loaded.firstID,
                "ram_bytes": ProcessInfo.processInfo.physicalMemory,
                "freed_bytes": freed.map { $0 as Any } ?? NSNull(), "note": unloadNote ?? NSNull(), "wait_seconds": Int(Self.unloadWait),
                "error": Self.stoppedFailure.wire,
            ]
        }
    }
    func stop() {
        // A memory search while idle ends with the server: its line says so now, whether or not its job gets to.
        if jobs.report?["state"] as? String == "running", let fresh = try? Settings.load(flags: flags) {
            Tuning.settleAutoMeasures(fresh)
        }
        lock.withLock {
            stopped = true
            launching?.terminate()
            // Wake bounded ingress reads and proxy disconnect detection. Closing/removing a descriptor uses this
            // same lock, so a recycled descriptor can never be shut down by an old client snapshot.
            for descriptor in clientDescriptors { _ = shutdown(descriptor, SHUT_RDWR) }
        }
        jobs.cancel()
    }
    /// Call outside the accept/request threads, after stopping the child scheduler. One bounded deadline covers
    /// remaining metric producers and the writer; a blocked sink keeps its bounded queue alive after false.
    @discardableResult func stopAndJoin(timeout: TimeInterval = 30) -> Bool {
        stop()
        guard timeout.isFinite, timeout >= 0 else { return false }
        let deadline = DispatchTime.now() + timeout
        guard requests.wait(timeout: deadline) == .success else { return false }
        let now = DispatchTime.now().uptimeNanoseconds
        let left = deadline.uptimeNanoseconds > now ? Double(deadline.uptimeNanoseconds - now) / 1e9 : 0
        return httpMetrics.flush(timeout: left)
    }
    private func retireClient(_ descriptor: Int32) {
        lock.withLock {
            clientDescriptors.remove(descriptor)
            close(descriptor)
        }
    }

    // MARK: children

    /// The child's command line: the front's own options, with this model, a free port and `--single-model`.
    private func arguments(_ row: Row, port: UInt16, trial: UInt64?) -> [String] {
        var arguments = ["serve"]
        for (key, values) in flags.sorted(by: { $0.key < $1.key })
        where key != "port" && key != "listen" && (row.spec == nil || key != "model") && (trial == nil || key != "memory-limit") {
            for value in values { arguments.append("--\(key)=\(value)") }
        }
        return arguments + ["--single-model", "--port", String(port), "--listen", lan ? "lan" : "local"]
            + (row.spec.map { ["--model", $0] } ?? []) + (trial.map { ["--memory-limit=\($0 / 1_000_000)MB"] } ?? [])
    }

    /// What can be known before the running model is stopped for `id` (`ModelScheduler`'s `ready`): it is listed, its
    /// drive is connected and readable, and its settings load. Returns its row.
    @discardableResult private func ready(_ id: String) throws -> Row {
        guard !isUnloaded else { throw Self.stoppedFailure }
        guard let row = rows().first(where: { $0.id == id }) else {
            throw ServingFailure("No model named \(id)", status: 404, code: "model_not_found")
        }
        // A model on a drive macOS has not let Origami read: a child would wait on the permission prompt forever (the
        // attempt here is what makes macOS show it), or, refused, take the model for missing; so say which drive instead
        // (the dashboard says how to allow it). The drive gets 30 s to wake first, and the menu learns what this found.
        let spec = row.spec ?? settings.model
        if let path = spec.map({ ModelRegistry.entry($0, settings: settings)?["path"] as? String ?? Settings.expand($0) }),
            path.hasPrefix("/Volumes/")
        {
            let state = Stores.check(path, within: 30)
            let drive = Stores.driveName(path)
            if state != .available { lock.withLock { checked[path] = (state, checked[path]?.bytes, Date()) } }
            if state == .notConnected {
                throw ServingFailure(
                    "\(drive), where \(id) is, is not connected: connect it, then reload", status: 503, type: "engine_unavailable")
            }
            if state == .noPermission || state == .slow {
                lock.withLock { permission = ["drive": drive, "path": path] }
                throw ServingFailure(
                    (state == .slow
                        ? "\(drive), where the model is, did not answer within 30 s; macOS may be waiting for permission: "
                        : "macOS has not let Origami read \(drive), where the model is: ")
                        + "allow Removable Volumes for \(PermissionSubject.name) in "
                        + "System Settings > Privacy & Security > Files & Folders, then reload", status: 503, type: "engine_unavailable"
                )
            }
        }
        _ = try Settings.load(flags: flags.merging(row.spec.map { ["model": [$0]] } ?? [:]) { $1 })
        return row
    }

    private func start(_ id: String) throws -> Child {
        guard !isUnloaded else { throw Self.stoppedFailure }
        do {
            lock.withLock {
                starting = id
                failure = nil
                permission = nil
                driveMisses = 0  // a new child opens the model's files afresh
            }
            let said = LastError()
            defer {
                lock.withLock {
                    starting = nil
                    startingLog = nil
                    launching = nil
                }
            }
            lock.withLock { startingLog = said }
            let row = try ready(id)
            let port = try Self.freePort()
            let process = Process()
            let errors = Pipe()
            // A test aid: a stand-in that answers as a loaded model (`Tests/dashboard-control-checks.py`).
            process.executableURL =
                ProcessInfo.processInfo.environment["ORIGAMI_TEST_WORKER"].map { URL(fileURLWithPath: $0) } ?? Bundle.main.executableURL
            let trial = lock.withLock { memoryTrial }
            process.arguments = arguments(row, port: port, trial: trial)
            if trial != nil {
                process.environment = ProcessInfo.processInfo.environment.merging([Tuning.trialVariable: "1"]) { $1 }
            }
            var launchFlags = flags.merging(row.spec.map { ["model": [$0]] } ?? [:]) { $1 }
            if let trial { launchFlags["memory-limit"] = ["\(trial / 1_000_000)MB"] }
            let launchSettings = try Settings.load(flags: launchFlags)
            let launchConfiguration = DashboardState.configuration(launchSettings)
            process.standardError = errors
            // A model process that crashes says nothing itself: the log names the signal (the front's own stops are
            // SIGTERM and SIGKILL). The next request starts it again.
            let note = log
            process.terminationHandler = { ended in
                guard ended.terminationReason == .uncaughtSignal, ![SIGTERM, SIGKILL, SIGINT].contains(ended.terminationStatus) else {
                    return
                }
                note(
                    "\(id) (process \(ended.processIdentifier)) crashed with signal \(ended.terminationStatus); the next request starts it again"
                )
            }
            try lock.withLock {
                guard !stopped else { throw ServingFailure("The server is stopping", status: 503, type: "engine_unavailable") }
                guard !unloaded else { throw Self.stoppedFailure }
                launching = process
                try process.run()
            }
            Thread.detachNewThread { Self.forward(errors.fileHandleForReading, keeping: said) }
            log(
                "starting \(id) (process \(process.processIdentifier))"
                    + (trial.map { " at a trial memory limit of \($0 / 1_000_000) MB" } ?? ""))
            // Until it listens: a Qwen child listens while it loads (requests wait there), the others once loaded.
            let deadline = Date().addingTimeInterval(Self.startWait)
            while process.isRunning {
                let probe = Self.connect(port: port)
                if probe >= 0 {
                    close(probe)
                    return Child(process, port: port, configuration: launchConfiguration)
                }
                guard Date() < deadline else {
                    // Stuck while loading: it holds memory the next start would need.
                    kill(process.processIdentifier, SIGKILL)
                    throw ServingFailure(
                        "\(id) did not start within \(Int(Self.startWait / 60)) minutes and was stopped"
                            + (said.recent.map { " (its last line: \($0))" } ?? "") + "; see the server log",
                        status: 503, type: "engine_unavailable")
                }
                usleep(200_000)
            }
            usleep(100_000)  // its last lines
            throw ServingFailure(
                said.line.map { "\(id) did not start: \($0)" } ?? "\(id) did not start (exit status \(process.terminationStatus))",
                status: 503, type: "engine_unavailable")
        } catch {
            // A start that Stop and unload ended (its child terminated) is the stop, not a failure to show.
            guard !isUnloaded else { throw Self.stoppedFailure }
            let failure = ServingFailure.from(error)
            lock.withLock { self.failure = failure.description }
            log(failure.description)
            throw failure
        }
    }

    /// The last error a child printed (`origami: …`, as `main` prints one), for the page when it does not start.
    /// Also its last line of any kind (`recent`, without the log's time and `origami: `) and when it started, for the
    /// loading state's progress.
    final class LastError: @unchecked Sendable {
        private let lock = NSLock()
        private var text: String?, latest: String?
        let started = Date()
        var line: String? {
            get { lock.withLock { text } }
            set { lock.withLock { text = newValue } }
        }
        var recent: String? {
            get { lock.withLock { latest } }
            set { lock.withLock { latest = newValue } }
        }
    }

    /// Copies a child's standard error to the server's, keeping its last error line and its last line.
    private static func forward(_ handle: FileHandle, keeping said: LastError) {
        while case let chunk = handle.availableData, !chunk.isEmpty {
            FileHandle.standardError.write(chunk)
            for line in String(decoding: chunk, as: UTF8.self).split(separator: "\n") {
                if line.hasPrefix("origami: ") { said.line = String(line.dropFirst("origami: ".count)) }
                let text = line.replacing(#/^\d{4}-\d\d-\d\dT[\d:.]+Z? /#, with: "").replacing(#/^origami: /#, with: "")
                if !text.trimmingCharacters(in: .whitespaces).isEmpty { said.recent = String(text.prefix(200)) }
            }
        }
    }

    /// The model being started: its last log line and the seconds since it began (`loading` in `/health`, `/v1/status`
    /// and the dashboard's state, and the 503 a request gets meanwhile), or nil.
    private func loadingProgress() -> [String: Any]? {
        guard let said = lock.withLock({ startingLog }) else { return nil }
        return ["seconds": Int(Date().timeIntervalSince(said.started)), "line": said.recent ?? NSNull()]
    }
    static func progressText(_ progress: [String: Any]?) -> String {
        guard let seconds = progress?["seconds"] as? Int else { return "" }
        return " (loading for \(seconds) s" + ((progress?["line"] as? String).map { "; last: " + $0 } ?? "") + ")"
    }

    /// SIGTERM: the child finishes its running request and its prompt-cache saves, as a one-model server does when
    /// stopped; SIGKILL after two minutes.
    private func stop(_ child: Child) {
        let process = child.process
        guard process.isRunning else { return }
        let footprint = ProcessMemory.footprint(process.processIdentifier)
        lock.withLock {
            held = footprint
            pendingReload = nil
        }
        process.terminate()
        let deadline = Date().addingTimeInterval(120)
        while process.isRunning && Date() < deadline { usleep(100_000) }
        if process.isRunning {
            kill(process.processIdentifier, SIGKILL)
            process.waitUntilExit()
        }
    }

    // MARK: requests

    func run(listener: Int32) {
        lock.lock()
        guard !stopped else {
            lock.unlock()
            return
        }
        requests.enter()
        lock.unlock()
        defer { requests.leave() }
        Thread.detachNewThread { [weak self] in
            var measured = Date()
            while true {
                // Every 5 s at most, so `idle_unload_minutes` set meanwhile counts (a quarter of it when shorter); the
                // memory measurement looks once per `AutoMeasure.every`.
                let unload = (self.flatMap { try? Settings.load(flags: $0.flags) })?.idleUnloadMinutes.map { $0 * 60 }
                Thread.sleep(forTimeInterval: min(AutoMeasure.every, 5, unload.map { max(0.25, $0 / 4) } ?? .infinity))
                guard let self, !self.isStopped else { return }
                self.checkModelDrive()
                if let unload { self.unloadIfIdle(after: unload) }
                if Date().timeIntervalSince(measured) >= AutoMeasure.every * 0.99 {
                    measured = Date()
                    self.measureIfIdle()
                }
            }
        }
        var poller = pollfd(fd: listener, events: Int16(POLLIN), revents: 0)
        while !isStopped {
            guard poll(&poller, 1, 250) > 0 else { continue }
            let client = accept(listener, nil, nil)
            guard client >= 0 else { continue }
            Self.configure(client)
            lock.lock()
            guard !stopped else {
                lock.unlock()
                close(client)
                continue
            }
            clientDescriptors.insert(client)
            lock.unlock()
            let lifecycle = httpMetrics.begin()
            lifecycle.fields(observationTags.merging(["front_instance": frontInstance]) { $1 })
            let peer = Self.peer(client)
            // A request whose body falls behind loses its buffer to another (`HTTPConnectionBudget`): its read side is shut,
            // so its receive returns, and it is answered 503.
            guard let admission = connectionBudget.accept(peer: peer, drop: { _ = shutdown(client, SHUT_RD) }) else {
                lifecycle.fields(["http_status": 503, "outcome": "connection_limit"])
                Self.reply(client, ServingFailure("Too many connections; try again in a moment", status: 503, type: "engine_unavailable"))
                retireClient(client)
                httpMetrics.finish(lifecycle, outcome: "connection_limit")
                continue
            }
            requests.enter()
            Thread.detachNewThread { [self] in
                defer {
                    // Released before the descriptor closes, so an eviction never shuts a reused descriptor.
                    connectionBudget.release(admission)
                    retireClient(client)
                    httpMetrics.finish(lifecycle, outcome: "closed")
                    requests.leave()
                }
                handle(client, peer: peer, admission: admission, lifecycle: lifecycle)
            }
        }
    }

    private func handle(_ client: Int32, peer: String?, admission: Int, lifecycle: HTTPRequestLifecycle) {
        // Errors on `/v1/messages` (and its count) in Anthropic's shape, as the model's own server writes them.
        var anthropic = false
        func reject(_ failure: ServingFailure) {
            lifecycle.fields(["http_status": failure.status, "outcome": "rejected", "error_code": failure.code])
            guard anthropic else { return Self.reply(client, failure) }
            let body = WireWriter(format: .anthropic, id: "", created: 0, model: "", includeUsage: false).error(failure)
            Self.reply(client, status: failure.status, body, extra: failure.code == "stopped" ? ["X-Should-Retry: false"] : [])
        }
        let busy = ServingFailure("Request buffers are busy; try again in a moment", status: 503, type: "server_busy")
        /// The client's address when it is another device (a trusted proxy's `X-Forwarded-For`, else the peer's), and
        /// whether the request is remote: another device, or this Mac's network name (a proxy, or a rebound name).
        func origin(_ headers: [String: String]) throws -> (who: String?, remote: Bool) {
            let forwarded = peer == nil && settings.trustedProxy ? try HTTPIngressPolicy.forwardedPeer(headers) : nil
            let remote = forwarded != nil || lan && !(peer == nil && HostGuard.isLoopback(headers["host"] ?? ""))
            return (forwarded ?? peer, remote)
        }
        func authenticate(target: String, headers: [String: String]) throws {
            let path = String(target.split(separator: "?", maxSplits: 1).first ?? "")
            let (who, remote) = try origin(headers)
            let keyless = LocalHTTPServer.isAsset(path: path) || LocalHTTPServer.isDashboard(path: path, headers: headers)
            if !keyless, let who, let wait = authLimiter.retryAfter(who) {
                throw ServingFailure(
                    "Too many wrong keys from this address; try again in \(wait) s", status: 429, type: "rate_limit_error",
                    code: "too_many_key_failures")
            }
            func refuse(_ message: String) -> ServingFailure {
                if let who { authLimiter.failed(who) }
                return ServingFailure(message, status: 401, type: "authentication_error", code: "invalid_api_key")
            }
            if remote && !keyless {
                let fresh = (try? Settings.load(flags: flags)) ?? settings
                if !(fresh.lanOpen && fresh.apiKey == nil)
                    && !(fresh.remoteKey.map { LocalHTTPServer.carriesKey($0, headers: headers) } ?? false)
                {
                    throw refuse("Other devices need this server's key: send Authorization: Bearer <key>, or paste it into the dashboard")
                }
            }
            if !keyless, let key = settings.apiKey, !LocalHTTPServer.carriesKey(key, headers: headers) {
                throw refuse("This server needs its API key: send Authorization: Bearer <key> or x-api-key: <key>")
            }
        }
        let request: BoundedHTTPRequest
        do {
            guard
                let received = try BoundedHTTPIngress.receive(
                    client, lan: lan,
                    headers: { target, headers, bytes in
                        try authenticate(target: target, headers: headers)
                        guard connectionBudget.reserveBody(admission, bytes: bytes) else { throw busy }
                    }, progress: { connectionBudget.received(admission, bytes: $0) })
            else {
                if connectionBudget.evicted(admission) { return reject(busy) }
                lifecycle.fields(["outcome": "disconnected_ingress"])
                return
            }
            request = received
            anthropic = received.target.hasPrefix("/v1/messages")
            lifecycle.mark("parsed")
            lifecycle.fields(["method": request.method, "request_bytes": request.body.count])
            guard
                connectionBudget.classify(
                    admission, control: request.body.isEmpty && HTTPIngressPolicy.isControl(method: request.method, target: request.target))
            else {
                throw ServingFailure("Too many active requests; try again in a moment", status: 503, type: "server_busy")
            }
            var ingressLimits = LocalHTTPConfiguration()
            ingressLimits.applyContextLimits(settings.context)
            try HTTPIngressPolicy.checkJSON(request.body, maximumStructureBytes: ingressLimits.maximumStructureBytes)
        } catch {
            reject(ServingFailure.from(error))
            return
        }
        let raw = BoundedHTTPIngress.forwarded(request, requestID: lifecycle.id)
        let path = String(request.target.split(separator: "?", maxSplits: 1).first ?? "")
        lifecycle.fields(["path": path])
        let remote = (try? origin(request.headers))?.remote ?? true
        // Validate structure before the override scanner can invoke Foundation's JSON parser.
        if remote, TuneOverrides.requested(in: request.body) {
            return reject(ServingFailure("origami_tune is taken only from this Mac", status: 403))
        }
        let rows = rows()
        let ids = rows.map(\.id)
        func answer(_ object: [String: Any], status: Int = 200) {
            lifecycle.fields(["http_status": status, "outcome": status < 400 ? "front_response" : "rejected"])
            Self.reply(client, status: status, object, head: request.method == "HEAD")
        }
        let cancelled = { [self] in lock.withLock { cancelling } }
        func pass(_ want: String?) {
            // `start_on_request`: a request to a stopped server starts the model, as `origami start` does, and waits for it.
            if isUnloaded, (try? Settings.load(flags: flags))?.startOnRequest == true, (try? resume()) != nil {
                log("a request arrived while stopped: starting the model (start_on_request)")
            }
            guard !isUnloaded else { return reject(Self.stoppedFailure) }
            let arrived = Date()
            autoMeasure.noteRequest()
            // An app's request during a memory search (not one of the search's own, which carry `origami_tune`): the saved
            // limit is loaded again before it is admitted, so it is never served at a trial limit nor met by the reload.
            if !TuneOverrides.requested(in: request.body) { endTrialForRequest() }
            // A stream that waits (a switch, a load, the queue) gets its headers and a keepalive every 10 s, as the model's
            // own server sends them; what fails after that arrives as a stream error event in the request's format.
            let format =
                (try? JSONSerialization.jsonObject(with: request.body) as? [String: Any])?["stream"] as? Bool == true
                ? ["/v1/chat/completions": WireFormat.openAI, "/v1/messages": .anthropic, "/v1/responses": .responses][path] : nil
            let waiting = format.map { _ in StreamWait(client) }
            // A stream writes all the time, so a client that half-closed after its request still gets it; a reply sent
            // only once complete takes an end of input as the client leaving (Ollama's paths stream unless told not to).
            let streamed =
                ((try? JSONSerialization.jsonObject(with: request.body) as? [String: Any])?["stream"] as? Bool) ?? path.hasPrefix("/api/")
            func fail(_ failure: ServingFailure) {
                guard let format, waiting?.finish() == true else { return reject(failure) }
                Self.streamError(client, failure, format: format, lifecycle: lifecycle)
            }
            do {
                lifecycle.mark("queue_start")
                // Waiting: once the stream's headers went out its keepalives tell; before, an end of input is leaving.
                let running = try scheduler.acquire(want, fallback: firstID) { waiting?.lost ?? Self.left(client) }
                defer { scheduler.release() }
                lifecycle.mark("queue_end")
                guard !isUnloaded else { return fail(Self.stoppedFailure) }  // stopped while it waited
                // `origami_time_limit` counts from the request's arrival here, not from the model's: the wait for a
                // switch, a load or the queue comes off it, and a limit that passed meanwhile is the 408 it promises.
                guard let sent = Self.limitFromArrival(raw, body: request.body, waited: Date().timeIntervalSince(arrived)) else {
                    return fail(ServingFailure.timeLimit())
                }
                let streaming = waiting?.finish() == true ? format : nil
                if streaming != nil { lifecycle.fields(["http_status": 200]) }
                Self.pipe(
                    sent, client: client, port: running.child.port, cancelled: cancelled, lifecycle: lifecycle, streaming: streaming,
                    endOfInputLeaves: !streamed)
            } catch is ModelScheduler<Child>.Gone {
                _ = waiting?.finish()
                lifecycle.fields(["outcome": "disconnected_queue"])
            } catch { fail(ServingFailure.from(error)) }
        }
        // Only this server's own page changes it: a JSON body forces a browser's cross-site preflight, and an Origin, when
        // sent, must be the page's own (`HostGuard.ownPage`, as the model's server checks it).
        func fromOwnPage() -> Bool {
            (request.headers["content-type"] ?? "").hasPrefix("application/json")
                && HostGuard.ownPage(origin: request.headers["origin"], host: request.headers["host"] ?? "")
        }
        // The page and its files come from the front, so they show while no model runs (none yet, or one that cannot start).
        let asset = DashboardPage.asset(path)
        if asset != nil || LocalHTTPServer.isDashboard(path: path, headers: request.headers) {
            guard request.method == "GET" || request.method == "HEAD" else {
                return reject(ServingFailure("Use GET for the dashboard", status: 405))
            }
            lifecycle.fields(["http_status": 200, "outcome": "front_response"])
            return Self.page(client, head: request.method == "HEAD", asset: asset)
        }
        switch path {
        case "/v1/dashboard":
            guard request.method == "GET" else { return reject(ServingFailure("Use GET for dashboard state", status: 405)) }
            let (state, status) = dashboardState()
            answer(state, status: status)
        case "/v1/tuning/apply":
            guard request.method == "POST", fromOwnPage() else { return reject(ServingFailure("Apply from this dashboard", status: 403)) }
            do {
                DashboardState.mutationLock.lock()
                defer { DashboardState.mutationLock.unlock() }
                answer(try applyTuning(request), status: 202)
            } catch { reject(ServingFailure.from(error)) }
        case "/v1/config/restore":
            guard request.method == "POST", fromOwnPage() else { return reject(ServingFailure("Restore from this dashboard", status: 403)) }
            do {
                DashboardState.mutationLock.lock()
                defer { DashboardState.mutationLock.unlock() }
                guard jobs.report?["state"] as? String != "running" else {
                    throw ServingFailure("Wait for the current operation before restoring settings", status: 409)
                }
                let fresh = try Settings.load(flags: flags)
                try DashboardState.check(request.headers["if-match"], settings: fresh)
                let saved = try Settings.readConfig(fresh.dataDir.appendingPathComponent("last-working.json"))
                guard let values = saved[fresh.model ?? ""] as? [String: String] else {
                    throw ServingFailure("No working configuration has been saved for this model", status: 404)
                }
                // One saved before `expert_bits` was retired restores without it.
                answer(try Commands.configWriter(fresh)(values.filter { $0.key != "expert_bits" }))
            } catch { reject(ServingFailure.from(error)) }
        // Presets: named copies of the Setup settings (`Presets`); a use saves its values, as a restore does.
        case _ where path == "/v1/presets" || path.hasPrefix("/v1/presets/"):
            let name = path == "/v1/presets" ? nil : String(path.dropFirst("/v1/presets/".count)).removingPercentEncoding ?? ""
            if request.method == "GET", name == nil {
                return answer(["presets": Presets.list((try? Settings.load(flags: flags)) ?? settings)])
            }
            guard fromOwnPage() else {
                return reject(ServingFailure("Presets are changed only from this server's own dashboard", status: 403))
            }
            do {
                DashboardState.mutationLock.lock()
                defer { DashboardState.mutationLock.unlock() }
                let fresh = try Settings.load(flags: flags)
                // A new preset keeps, and a use replaces, the settings the page showed.
                if request.method == "POST" { try DashboardState.check(request.headers["if-match"], settings: fresh) }
                if request.method == "POST", name != nil, jobs.report?["state"] as? String == "running" {
                    throw ServingFailure("Wait for the current operation before using a preset", status: 409)
                }
                let body = (try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any] ?? [:]
                answer(try Presets.change(request.method, name, body, settings: fresh))
            } catch { reject(ServingFailure.from(error)) }
        // Access: who reaches this server, the addresses other devices open, and (to this Mac only) their key and its QR code.
        case "/v1/lan":
            guard request.method == "GET" else { return reject(ServingFailure("Use GET for access", status: 405)) }
            answer(access(remote: remote))
        // Applies `listen` (the socket is bound once): launchd stops this server, its requests finishing, and starts it again.
        case "/v1/restart":
            guard request.method == "POST" else { return reject(ServingFailure("Use POST to restart", status: 405)) }
            guard fromOwnPage() else { return reject(ServingFailure("Restart from this server's own page", status: 403)) }
            guard Access.launchdLabel != nil else {
                return reject(ServingFailure("Origami was not started by launchd: restart origami serve to apply it", status: 409))
            }
            answer(["restarting": true], status: 202)
            log("restarting to apply the saved settings (from the dashboard)")
            do { try Access.restart() } catch { log("restart: \(error)") }
        // Stop and unload, and Start (`unload`, `resume`): the dashboard's button and `origami stop` / `origami start`.
        case "/v1/stop", "/v1/start":
            guard request.method == "POST" else { return reject(ServingFailure("Use POST to stop or start Origami", status: 405)) }
            guard fromOwnPage() else {
                return reject(
                    ServingFailure(
                        "Stop and start Origami from this server's own page, or with origami stop and origami start", status: 403))
            }
            do {
                if path == "/v1/stop" {
                    guard jobs.report?["state"] as? String != "running" else {
                        throw ServingFailure("Cancel the current operation before stopping", status: 409)
                    }
                    answer(try unload(), status: 202)
                } else {
                    let (status, body) = try resume()
                    answer(body, status: status)
                }
            } catch { reject(ServingFailure.from(error)) }
        case "/v1/reload":
            guard request.method == "POST" else { return reject(ServingFailure("Use POST to reload", status: 405)) }
            guard fromOwnPage() else { return reject(ServingFailure("Reload from this server's own page", status: 403)) }
            do {
                guard jobs.report?["state"] as? String != "running" || jobs.report?["kind"] as? String == "tune" else {
                    throw ServingFailure("Wait for the current operation before reloading", status: 409)
                }
                // `origami tune --memory` only: a limit to load at, kept in this process, never saved.
                let body = (try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any]
                let trial = (body?["memory_limit_trial"] as? NSNumber)?.uint64Value
                // A search from a terminal (no job) ends with this restore: the next trial is a new search's.
                if body?["restore_memory"] as? Bool == true, jobs.report?["state"] as? String != "running" {
                    lock.withLock { trialEnded = false }
                }
                // Its restore at the end: nothing to do when no trial is loaded (a request already restored the limit).
                if body?["restore_memory"] as? Bool == true, lock.withLock({ memoryTrial == nil }) {
                    return answer(["reloading": false, "model": scheduler.current?.id ?? firstID])
                }
                if let trial,
                    trial == 0 || trial > WiredMemoryPolicy.maximumBudgetBytes(physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)
                {
                    throw ServingFailure("memory_limit_trial is more than this Mac allows")
                }
                answer(reloading(try reload(trial: trial)), status: 202)
            } catch { reject(ServingFailure.from(error)) }
        case let route where ["/v1/config", "/v1/memory/wired"].contains(route) && request.method == "POST":
            // Saved by the front, so the dashboard's Apply works while no model runs (none yet, or one that cannot start).
            guard fromOwnPage() else {
                return reject(ServingFailure("Settings are saved only from this server's own dashboard", status: 403))
            }
            guard let values = (try? JSONSerialization.jsonObject(with: request.body)) as? [String: String], !values.isEmpty else {
                return reject(ServingFailure("POST a JSON object of settings with text values, such as {\"context\": \"32k\"}"))
            }
            do {
                DashboardState.mutationLock.lock()
                defer { DashboardState.mutationLock.unlock() }
                answer(try saveSettings(values, path: path, request: request))
            } catch let conflict as DashboardState.Conflict {
                answer(conflict.body, status: 409)
            } catch { reject(ServingFailure.from(error)) }
        // The Experts card's hot copy: the running model builds and attaches it while serving (the request goes to it); a
        // further Qwen copy on another drive (`copy`) is built by `origami setup` as a job, after which the model reloads;
        // with no model running the change is saved.
        case "/v1/hot-copy":
            guard jobs.report?["state"] as? String != "running", !scheduler.isDraining else {
                return reject(ServingFailure("Wait for the current operation", status: 409))
            }
            do { try DashboardState.check(request.headers["if-match"], settings: try Settings.load(flags: flags)) } catch {
                return reject(ServingFailure.from(error))
            }
            guard request.method == "POST" else { return reject(ServingFailure("Use POST to change the hot copy", status: 405)) }
            guard fromOwnPage() else {
                return reject(ServingFailure("The hot copy is changed only from this server's own dashboard", status: 403))
            }
            guard let body = (try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any] else {
                return reject(ServingFailure("POST {\"build\": true}, {\"drive\": \"/Volumes/<drive>\"} or {\"size\": \"40GB\"}"))
            }
            if !isUnloaded, let running = scheduler.acquireRunning() {
                defer { scheduler.releaseRunning() }
                if rows.first(where: { $0.id == running.id })?.family != .qwen || body["copy"] == nil {
                    return Self.pipe(raw, client: client, port: running.child.port, cancelled: cancelled, lifecycle: lifecycle)
                }
            }
            do {
                let fresh = try Settings.load(flags: flags)
                let saved = try HotCopy.save(body, settings: fresh, model: FamilySetup.gguf(fresh), qwen: FamilySetup.gguf(fresh) == nil)
                guard scheduler.current != nil, FamilySetup.gguf(fresh) == nil, body["copy"] != nil else {
                    return answer(["saved": saved, "restart": false])
                }
                // A further Qwen copy: setup builds it where the settings say, then the model reloads if that changed what
                // it reads.
                let before = Stores.signature(fresh)
                try jobs.start(
                    kind: "hot-copy", title: "Copying the experts", arguments: ["setup", "--yes"]
                ) { [weak self] state in self?.afterHotCopy(state, before: before) }
                answer(["saved": saved, "job": "hot-copy", "reload": "if the job changes the copy, " + Self.pendingText], status: 202)
            } catch { reject(ServingFailure.from(error)) }
        // Forgets the learned memory budget: the running model's own history, else the file.
        case "/v1/memory/learned":
            guard request.method == "POST" else {
                return reject(ServingFailure("Use POST to forget the learned memory budget", status: 405))
            }
            guard fromOwnPage() else { return reject(ServingFailure("This is changed only from this server's own dashboard", status: 403)) }
            if !isUnloaded, let running = scheduler.acquireRunning() {
                defer { scheduler.releaseRunning() }
                return Self.pipe(raw, client: client, port: running.child.port, cancelled: cancelled, lifecycle: lifecycle)
            }
            Tuning.memoryHistory(settings).reset()
            answer(["reset": true])
        case "/v1/jobs":
            if request.method == "GET" { return answer(["job": jobs.report ?? NSNull()]) }
            guard ["POST", "DELETE"].contains(request.method) else {
                return reject(ServingFailure("Use GET, POST or DELETE for jobs", status: 405))
            }
            guard fromOwnPage() else { return reject(ServingFailure("Start jobs from this server's own page", status: 403)) }
            if request.method == "DELETE" {
                jobs.cancel()
                return answer(["job": jobs.report ?? NSNull()])
            }
            do {
                let body = (try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any] ?? [:]
                if let ask = try addQuestion(body) { return answer(["confirm": ask]) }
                try startJob(body)
                answer(["job": jobs.report ?? NSNull()], status: 202)
            } catch { reject(ServingFailure.from(error)) }
        case "/v1/models/found":
            guard request.method == "GET" else { return reject(ServingFailure("Use GET for found models", status: 405)) }
            answer(found(fresh: formQueryItems(request.target).contains { $0.name == "fresh" }))
        case "/v1/models/check":
            guard request.method == "GET" else { return reject(ServingFailure("Use GET to check a model", status: 405)) }
            let items = formQueryItems(request.target)
            let source = items.first { $0.name == "source" }?.value ?? ""
            answer(
                check(
                    source.trimmingCharacters(in: .whitespaces),
                    to: items.first { $0.name == "to" }?.value.flatMap { $0.isEmpty ? nil : $0 }))
        // The folder browser reads the settings file afresh: a folder saved from the page shows at once, before a reload.
        case "/v1/fs":
            guard request.method == "GET" else { return reject(ServingFailure("Use GET to list a folder", status: 405)) }
            let items = formQueryItems(request.target)
            let fresh = (try? Settings.load(flags: flags)) ?? settings
            do {
                if let path = items.first(where: { $0.name == "path" })?.value, !path.isEmpty {
                    answer(
                        try Folders.list(
                            path, models: items.contains { $0.name == "models" }, settings: fresh,
                            search: items.first { $0.name == "search" }?.value ?? "",
                            offset: Int(items.first { $0.name == "offset" }?.value ?? "0") ?? 0))
                } else {
                    answer(Folders.places(fresh))
                }
            } catch {
                let failure = ServingFailure.from(error)
                answer(["error": failure.wire], status: failure.status)
            }
        case "/v1/fs/mkdir":
            guard request.method == "POST" else { return reject(ServingFailure("Use POST to make a folder", status: 405)) }
            guard fromOwnPage() else { return reject(ServingFailure("Folders are made only from this server's own page", status: 403)) }
            do {
                guard let path = ((try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any])?["path"] as? String else {
                    throw ServingFailure("POST {\"path\": \"/full/path/of/the/new/folder\"}")
                }
                answer(try Folders.makeFolder(path, settings: (try? Settings.load(flags: flags)) ?? settings))
            } catch {
                let failure = ServingFailure.from(error)
                answer(["error": failure.wire], status: failure.status)
            }
        case "/v1/storage/plan":
            guard request.method == "POST", fromOwnPage() else { return reject(ServingFailure("Preview from this dashboard", status: 403)) }
            do {
                answer(try storagePlan(request))
            } catch { reject(ServingFailure.from(error)) }
        case "/v1/storage":
            guard request.method == "GET" else { return reject(ServingFailure("Use GET for the storage map", status: 405)) }
            let fresh = (try? Settings.load(flags: flags)) ?? settings
            let before = DashboardState.revision(fresh)
            var active: String?
            var sources: [[String: Any]]?
            if let running = scheduler.acquireRunning() {
                active = running.id
                let state = DashboardState.status(port: running.child.port, key: settings.apiKey)
                sources = (state["experts_from"] as? [String: Any])?["sources"] as? [[String: Any]]
                scheduler.releaseRunning()
            }
            let report = Storage.report(fresh, running: active, found: foundForStorage(fresh), sources: sources)
            if before != DashboardState.revision(fresh) { answer(["retry": true], status: 409) } else { answer(report) }
        // The saved prompts are the running model's: never a reason to start or switch one.
        case _ where path == "/v1/prompts" || path.hasPrefix("/v1/prompts/"):
            guard !isUnloaded else { return reject(Self.stoppedFailure) }
            guard let running = scheduler.acquireRunning() else {
                return reject(
                    ServingFailure("No model is running; its saved prompts show once one runs", status: 503, type: "engine_unavailable"))
            }
            defer { scheduler.releaseRunning() }
            Self.pipe(raw, client: client, port: running.child.port, cancelled: cancelled, lifecycle: lifecycle)
        case "/v1/models", "/api/tags", "/api/version":
            guard request.method == "GET" else { return reject(ServingFailure("Use GET for models", status: 405)) }
            // `?all`: every model whatever `list_models` says (the dashboard's model picker).
            let all = formQueryItems(request.target).contains { $0.name == "all" }
            answer(listing(path, rows: rows, all: all || listsAll, recheck: all))
        case _ where path.hasPrefix("/v1/models/"):
            guard request.method == "GET" else { return reject(ServingFailure("Use GET for models", status: 405)) }
            let id = String(path.dropFirst("/v1/models/".count)).removingPercentEncoding ?? ""
            // `origami` (the setup's name) and every model's own id with its `-<level>` names.
            let entries = [false, true].flatMap { listing("/v1/models", rows: rows, all: $0)["data"] as? [[String: Any]] ?? [] }
            guard let entry = entries.first(where: { $0["id"] as? String == id }) else {
                return reject(ServingFailure("No model named \(id)", status: 404, code: "model_not_found"))
            }
            answer(entry)
        case "/api/show" where request.method == "POST" && !listsAll:
            pass(firstID)
        case "/api/show" where request.method == "POST":
            let body = (try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any]
            let id = ModelScheduler<Child>.resolve((body?["model"] ?? body?["name"]) as? String, ids: ids)
            guard let id, id != scheduler.current?.id, let row = rows.first(where: { $0.id == id }) else { return pass(id) }
            let (details, abilities) = LocalHTTPServer.ollamaDescription(
                row.family, quantization: row.quantization, vision: Self.vision(row))
            let architecture = details["family"] as? String ?? ""
            answer([
                "modelfile": "", "parameters": "", "template": "", "details": details,
                "model_info": ["general.architecture": architecture, architecture + ".context_length": settings.context],
                "capabilities": abilities,
            ])
        default:
            if request.method == "POST" {
                // One listed model: every request runs the setup, whatever name it sends.
                guard listsAll else { return pass(firstID) }
                let body = (try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any]
                let name = body?["model"] as? String
                let id = ModelScheduler<Child>.resolve(name, ids: ids)
                // A name not listed is not answered by whichever model runs under that name; no name, or `origami`, is.
                if id == nil, let name, !name.isEmpty, ![Self.setupID, Self.setupID + ":latest"].contains(name) {
                    return reject(ServingFailure("No model named \(name)", status: 404, code: "model_not_found"))
                }
                return pass(id)
            }
            // Stopped: the front answers health and status itself (503, as apps should not send requests), the rest refused.
            if isUnloaded {
                switch path {
                case "/api/ps": return answer(["models": [Any]()])
                case "/health", "/v1/health", "/v1/status", "/":
                    var state = stoppedState()
                    markSlowed(&state)
                    return answer(state, status: 503)
                default: return reject(Self.stoppedFailure)
                }
            }
            // Health, status and the dashboard go to whichever model runs, without waiting behind a switch; health and
            // status add what slows it now (`slowed_by`) and, while a reload waits for the running reply, say so (`pending`).
            if let running = scheduler.acquireRunning() {
                defer { scheduler.releaseRunning() }
                let pending = lock.withLock { pendingReload }
                if ["/health", "/v1/health", "/v1/status", "/"].contains(path) {
                    var state = DashboardState.status(port: running.child.port, key: settings.apiKey, path: path)
                    if let pending { state["pending"] = pending }
                    markDegraded(&state)
                    markSlowed(&state, port: path == "/v1/status" ? nil : running.child.port)
                    return answer(state, status: state["status"] as? String == "unavailable" ? 503 : 200)
                }
                return Self.pipe(raw, client: client, port: running.child.port, cancelled: cancelled, lifecycle: lifecycle)
            }
            let (starting, failure) = lock.withLock { (self.starting, self.failure) }
            switch path {
            case "/api/ps": answer(["models": [Any]()])
            case "/health", "/v1/health", "/v1/status", "/":
                let permission = lock.withLock { self.permission }
                var state: [String: Any] =
                    failure != nil && starting == nil
                    ? [
                        "status": "failed", "failure": failure ?? "", "version": version, "model": firstID,
                        "permission": permission ?? NSNull(), "download": Models.defaultRepo,
                    ]
                    : ["status": "loading", "version": version, "model": starting ?? firstID, "loading": loadingProgress() ?? NSNull()]
                markSlowed(&state)
                answer(state, status: failure != nil && starting == nil ? 503 : 200)
            default:
                reject(
                    ServingFailure(
                        "The model is starting\(Self.progressText(loadingProgress())); try again in a moment", status: 503,
                        type: "engine_unavailable"))
            }
        }
    }

    /// `/v1/dashboard`: the page's state, or a retry when the settings changed while it was read.
    private func dashboardState() -> (state: [String: Any], status: Int) {
        let fresh = (try? Settings.load(flags: flags)) ?? settings
        let before = DashboardState.revision(fresh)
        var (state, active, generation) = liveState()
        state["front_http"] = httpMetrics.snapshot
        state["front_connections"] = connectionBudget.snapshot
        let revision = DashboardState.revision(fresh)
        guard before == revision else { return (["retry": true], 409) }
        let job = jobs.report
        let available = state["status"] as? String == "ready"
        let queue = state["queue"] as? [String: Any] ?? [:]
        let idle =
            available && queue["busy"] as? Bool != true && (queue["queued"] as? Int ?? 0) == 0 && job?["state"] as? String != "running"
        let runningConfiguration = scheduler.current?.child.configuration ?? [:]
        let reloadRequired = reloadRequired(fresh, state: state)
        let pending =
            reloadRequired
            ? DashboardState.pending(
                fresh, running: runningConfiguration, runningModel: scheduler.current?.id ?? "", servedModel: Self.servedID(fresh),
                state: state) : ([], [:])
        let tuneBlock = DashboardState.tuneBlock(state: state, job: job, reloadRequired: reloadRequired)
        state["dashboard"] = [
            "revision": revision, "generation": dashboardSession + ":" + generation,
            "observed_at": Date().timeIntervalSince1970, "host": ProcessInfo.processInfo.hostName,
            "configured_model": fresh.model ?? "", "active_model": active ?? NSNull(), "saved_settings": fresh.report,
            "wired_memory": DashboardState.wiredMemory(fresh, state: state),
            "job": job ?? NSNull(), "can_tune": tuneBlock.isEmpty, "can_measure": idle, "tune_block": tuneBlock,
            "reload_required": reloadRequired, "pending_changes": pending.0, "pending_revert": pending.1,
            "can_restore":
                ((try? Settings.readConfig(fresh.dataDir.appendingPathComponent("last-working.json")))?[fresh.model ?? ""] != nil),
            "presets": Presets.list(fresh), "last_tuning": DashboardState.lastRun(fresh),
            "tuning_proposal": (try? Settings.readConfig(Tuning.proposalFile(fresh))) ?? [:],
        ]
        if scheduler.current?.id == Self.servedID(fresh), scheduler.current?.child.configuration == DashboardState.configuration(fresh) {
            DashboardState.checkpoint(fresh, status: state)
        }
        return (state, state["status"] as? String == "unavailable" ? 503 : 200)
    }

    /// Whether the loaded model runs other settings than the saved ones; a reload already pending is said by `pending`,
    /// not asked for again. The dashboard, a save and the start of a tuning job all ask this.
    private func reloadRequired(_ fresh: Settings, state: [String: Any]) -> Bool {
        state["pending"] == nil
            && (scheduler.current.map {
                $0.id != Self.servedID(fresh) || $0.child.configuration != DashboardState.configuration(fresh)
                    || DashboardState.memoryReloadRequired(fresh, state: state)
            } ?? false)
    }

    /// The model's state as `/v1/dashboard` reports it: stopped, the running model's status, or starting or failed.
    private func liveState() -> (state: [String: Any], active: String?, generation: String) {
        var state: [String: Any]
        var generation = "none"
        var active: String?
        if isUnloaded {
            state = stoppedState()
        } else if let running = scheduler.acquireRunning() {
            generation = String(running.child.process.processIdentifier)
            active = running.id
            state = DashboardState.status(port: running.child.port, key: settings.apiKey)
            scheduler.releaseRunning()
            markDegraded(&state)
        } else {
            let (starting, failure, permission) = lock.withLock { (self.starting, self.failure, self.permission) }
            state = [
                "status": failure != nil && starting == nil ? "failed" : "loading", "version": version,
                "model": starting ?? firstID, "failure": failure ?? NSNull(), "loading": loadingProgress() ?? NSNull(),
            ]
            // As `/v1/status` says them: the page's first run offers the download, and names a drive macOS has not allowed.
            if failure != nil && starting == nil {
                state["permission"] = permission ?? NSNull()
                state["download"] = Models.defaultRepo
            }
        }
        if let pending = lock.withLock({ pendingReload }), active != nil { state["pending"] = pending }
        markSlowed(&state)
        return (state, active, generation)
    }

    /// `slowed_by` (`SlowedBy`) from `state` with what the front knows itself, added to `state` too: this Mac's heat and
    /// Low Power Mode (`mac_now`), the largest other apps while memory is short of what Origami may use
    /// (`memory_holders`); and read only: the job and the last tuning run while none is applied.
    /// `port`: `state` is the model's `/health`, so the list comes from its `/v1/status` (one from the last 5 s is reused).
    private func markSlowed(_ state: inout [String: Any], port: UInt16? = nil) {
        if let port {
            if let last = lock.withLock({ lastSlowed }), Date().timeIntervalSince(last.at) < 5 {
                state["slowed_by"] = last.list
                return
            }
            var full = DashboardState.status(port: port, key: settings.apiKey)
            full["pending"] = state["pending"]
            markDegraded(&full)
            markSlowed(&full)
            state["slowed_by"] = full["slowed_by"]
            return
        }
        state["mac_now"] = [
            "thermal_state": ProcessInfo.processInfo.thermalState.rawValue, "low_power": ProcessInfo.processInfo.isLowPowerModeEnabled,
        ]
        if SlowedBy.memoryShort(state) { state["memory_holders"] = ProcessMemory.others().map { ["name": $0.name, "bytes": $0.bytes] } }
        var signals = state
        if let job = jobs.report { signals["job"] = job }
        if (state["tuning"] as? [String: Any])?["state"] as? String == "none" { signals["last_tuning"] = DashboardState.lastRun(settings) }
        let list = lock.withLock {
            let list = SlowedBy.stamp(SlowedBy.list(signals), seen: &slowSeen)
            lastSlowed = (Date(), list)
            return list
        }
        state["slowed_by"] = list
    }

    /// `/v1/tuning/apply` (under the caller's settings lock): saves the tuning result, then reloads.
    private func applyTuning(_ request: BoundedHTTPRequest) throws -> [String: Any] {
        let fresh = try Settings.load(flags: flags)
        let proposal = try Settings.readConfig(Tuning.proposalFile(fresh))
        try DashboardState.check(request.headers["if-match"], settings: fresh)
        guard let revision = proposal["revision"] as? String else {
            throw ServingFailure("Run tuning before applying a result", status: 409)
        }
        try DashboardState.check(revision, settings: fresh)
        guard jobs.report?["state"] as? String != "running", let running = scheduler.acquireRunning() else {
            throw ServingFailure("Wait for an idle loaded model", status: 409)
        }
        let state = DashboardState.status(port: running.child.port, key: settings.apiKey)
        scheduler.releaseRunning()
        let queue = state["queue"] as? [String: Any] ?? [:]
        guard state["status"] as? String == "ready", running.id == Self.servedID(fresh), queue["busy"] as? Bool != true,
            (queue["queued"] as? Int ?? 0) == 0
        else { throw ServingFailure("Wait for the configured model to be idle", status: 409) }
        guard let key = proposal["key"] as? String, key == (state["tuning"] as? [String: Any])?["key"] as? String else {
            throw ServingFailure("This result belongs to another model or context", status: 409)
        }
        if proposal["kind"] as? String == "performance", let entry = proposal["entry"] as? [String: Any] {
            try Tuning.save(fresh, key: key, entry: entry)
        } else {
            let choice = ((try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any])?["quality"] as? String ?? ""
            guard QualityDial.levels.contains(where: { $0.name == choice }),
                let result = (proposal["results"] as? [[String: Any]] ?? []).first(where: { $0["quality"] as? String == choice }),
                (result["samples"] as? Int ?? 0) >= 2,
                let rate = result["decode_tps"] as? Double, rate.isFinite, rate > 0,
                let first = result["first_token_seconds"] as? Double, first.isFinite, first >= 0,
                let spread = result["spread"] as? Double, spread.isFinite, spread >= 0
            else {
                throw ServingFailure("Choose a mode with at least two completed measurements")
            }
            _ = try Commands.configWriter(fresh)(["quality": choice])
        }
        try? FileManager.default.removeItem(at: Tuning.proposalFile(fresh))
        return reloading(try reload(), ["saved": true])
    }

    /// `/v1/lan`: who reaches this server, the addresses other devices open, and (to this Mac only) their key.
    private func access(remote: Bool) -> [String: Any] {
        let fresh = (try? Settings.load(flags: flags)) ?? settings
        let urls = Access.urls(port: port)
        var body: [String: Any] = [
            "listen": fresh.listenLAN ? "lan" : "local", "listening": lan ? "lan" : "local", "port": port, "urls": urls,
            "key_required": !(fresh.lanOpen && fresh.apiKey == nil),
            "key_setting": fresh.apiKey != nil ? "api_key" : "lan_key", "restart": Access.launchdLabel != nil,
        ]
        if !remote, let key = fresh.remoteKey {
            body["key"] = key
            if lan, let url = urls.first {
                body["qr"] =
                    Access.qr(url + "#key=" + (key.addingPercentEncoding(withAllowedCharacters: .alphanumerics) ?? key)) ?? NSNull()
            }
        } else if !remote, lan, fresh.remoteKey == nil, let url = urls.first {
            body["qr"] = Access.qr(url) ?? NSNull()
        }
        return body
    }

    /// `/v1/config` and `/v1/memory/wired` (under the caller's settings lock): the saved settings and what they need.
    private func saveSettings(_ values: [String: String], path: String, request: BoundedHTTPRequest) throws -> [String: Any] {
        let fresh = try Settings.load(flags: flags)
        try DashboardState.check(base: request.headers["if-match"], writing: Array(values.keys), settings: fresh)
        guard jobs.report?["state"] as? String != "running" else {
            throw ServingFailure("Wait for the current operation before saving settings", status: 409)
        }
        var result = try Commands.configWriter(fresh, coordinateWiredMemory: path == "/v1/memory/wired")(values)
        let saved = try Settings.load(flags: flags)
        result["revision"] = DashboardState.revision(saved)
        var state: [String: Any] = [:]
        if let running = scheduler.acquireRunning() {
            state = DashboardState.status(port: running.child.port, key: settings.apiKey)
            scheduler.releaseRunning()
        }
        result["wired_memory"] = DashboardState.wiredMemory(saved, state: state)
        result["reload_required"] = reloadRequired(saved, state: state)
        return result
    }

    /// `/v1/storage/plan`: what moving, deleting or forgetting an item would do.
    private func storagePlan(_ request: BoundedHTTPRequest) throws -> [String: Any] {
        let body = (try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any] ?? [:]
        let fresh = try Settings.load(flags: flags)
        // Remove unusable saved prompts: each folder that goes, with its size and the kept conversations that need a tick.
        if body["kind"] as? String == "prune" {
            let going = Storage.map(fresh, found: foundForStorage(fresh)).prompts.filter {
                if case .unusable = $0.state { true } else { false }
            }
            guard !going.isEmpty else { throw ServingFailure("Only an item nothing uses is removed", status: 409) }
            return [
                "kind": "prune", "name": "unusable saved prompts", "folders": going.map(\.json),
                "bytes": going.reduce(UInt64(0)) { $0 + $1.bytes }, "revision": DashboardState.revision(fresh),
            ]
        }
        let all = Storage.items(fresh, found: foundForStorage(fresh))
        guard let item = all.first(where: { $0.path == body["path"] as? String }) else {
            throw ServingFailure("This item is no longer available", status: 409)
        }
        let action = body["kind"] as? String ?? ""
        var preview: [String: Any] = [
            "kind": action, "path": item.path, "name": item.name, "role": item.role, "bytes": item.bytes,
            "allocated_bytes": item.allocated, "in_use": Storage.inUse(item, running: scheduler.current?.id),
            "parts": item.pieces.map(\.from), "model": item.model ?? NSNull(),
        ]
        if action == "move" {
            guard let to = body["to"] as? String else { throw ServingFailure("Choose a destination") }
            let move = try Storage.plan(item, to: to, settings: fresh, items: all)
            preview["to"] = move.folder
            preview["copy_bytes"] = move.renames ? 0 : move.bytes
            preview["required_bytes"] = move.renames ? 0 : Models.conversionNeeds(move.bytes)
            preview["free_bytes"] = Folders.space(move.folder)?.free ?? 0
        } else if action == "delete" {
            guard item.readable, item.deleteBlocked == nil else {
                throw ServingFailure(item.deleteBlocked ?? "Reconnect this disk first", status: 409)
            }
            preview["disables_cache"] = item.role == "hot copy"
        } else if !["remove", "forget"].contains(action) {
            throw ServingFailure("Unknown storage action")
        }
        preview["revision"] = DashboardState.revision(fresh)
        return preview
    }

    // MARK: the dashboard's model search and jobs

    /// An add with `"base"` (`model add <experts.gguf> --base`) checks the pairing before its job starts, writing nothing
    /// (`ExpertVariant.check`): a refusal is thrown with its reason, and what an experts-only file cannot show is returned
    /// for the page to ask about, until the add comes again with `"acknowledged": true`. Nil: start the job.
    private func addQuestion(_ body: [String: Any]) throws -> String? {
        guard body["kind"] as? String == "add", let source = body["source"] as? String, let base = body["base"] as? String,
            !base.isEmpty, !base.hasPrefix("-"), !source.hasPrefix("-")
        else { return nil }  // `startJob` refuses what is malformed
        let fresh = (try? Settings.load(flags: flags)) ?? settings
        let ask: String?
        do {
            ask = try ExpertVariant.check(
                experts: URL(fileURLWithPath: Settings.expand(source.trimmingCharacters(in: .whitespaces))), base: base, settings: fresh)
        } catch {
            throw ServingFailure(
                error is CLIError || error is EngineError ? "\(error)" : ServingFailure.from(error).description, status: 409)
        }
        return body["acknowledged"] as? Bool == true ? nil : ask
    }

    /// A memory trial still set once its search has ended: the model reloads with the saved settings (not while the
    /// server stops or is stopped; a stop clears the trial itself).
    /// It returns once that reload, or one already under way (the search's own restore), is done, so what is said after
    /// it is the final state.
    private func endMemoryTrial() {
        guard !isStopped else { return }
        if lock.withLock({ memoryTrial != nil }) {
            _ = try? reload(because: "to restore the saved memory limit after the memory search")
        }
        let deadline = Date().addingTimeInterval(Self.startWait + 150)
        while !isStopped, Date() < deadline, lock.withLock({ reloadsUnderway > 0 }) { usleep(100_000) }
    }

    /// An app's request while a trial limit is loaded: the trial ends here and the saved limit loads before the request
    /// goes on (after the search's running reply, as any reload waits for it); the search, seeing the model restarted,
    /// stops, and may not load another trial until its job ends (`trialEnded`).
    private func endTrialForRequest() {
        // Every request passes the lock: one that comes while another restores the limit (the trial cleared, the
        // scheduler's reload not yet begun) waits for it rather than being admitted to the trial-limit model.
        trialEnd.withLock {
            let running = scheduler.current?.id  // before `lock`, as `reload` reads it
            guard
                let id = lock.withLock({ () -> String? in
                    guard memoryTrial != nil else { return nil }
                    memoryTrial = nil
                    trialEnded = true
                    pendingReload = Self.pendingText + " to restore the saved memory limit: a request arrived"
                    reloadsUnderway += 1
                    defer { trialModel = nil }
                    return trialModel ?? running ?? loaded.firstID
                })
            else { return }
            log("a request arrived during the memory search: reloading \(id) at the saved memory limit before answering it")
            do { try scheduler.reload(id) } catch { log("reload: \(ServingFailure.from(error).description)") }
            lock.withLock {
                pendingReload = nil
                reloadsUnderway -= 1
            }
        }
    }

    /// The tuning key the running model looks its measured memory limit up by (a move or refresh changes it), or nil.
    private func runningTuningKey() -> String? {
        guard let running = scheduler.acquireRunning() else { return nil }
        defer { scheduler.releaseRunning() }
        return (DashboardState.status(port: running.child.port, key: settings.apiKey)["tuning"] as? [String: Any])?["key"] as? String
    }

    /// `memory_auto_measure` (`AutoMeasure`): the memory search started by itself when the saved settings, the job,
    /// the power and the last request allow it, for the model that is loaded (its own record and back-off: `memory-auto@`
    /// its name and context); `startJob` refuses it while that model is not idle and ready. Why it waits is kept with
    /// its time (`Tuning.noteAutoMeasureWaiting`).
    private func measureIfIdle() {
        guard !isUnloaded, let saved = try? Settings.load(flags: flags), saved.memoryAutoMeasure else { return }
        let fresh = modelSettings(saved)
        let conditions = AutoMeasure.Conditions(
            enabled: fresh.memoryAutoMeasure, limitSet: fresh.memoryLimit != nil,
            jobRunning: jobs.report?["state"] as? String == "running",
            measurement: runningTuningKey().flatMap { Tuning.memoryAdvice(fresh, key: $0, history: Tuning.memoryHistory(fresh)) },
            power: DriveSpeed.refusal(),
            lastTry: (Tuning.autoMeasureRecord(fresh)?["date"] as? String).flatMap { ISO8601DateFormatter().date(from: $0) })
        autoMeasure.tick(
            conditions,
            start: { ended in
                try startJob(["kind": "tune", "mode": "memory"]) { [jobs, unowned self] state in
                    let after = modelSettings((try? Settings.load(flags: flags)) ?? saved)
                    let advice = runningTuningKey().flatMap { Tuning.memoryAdvice(after, key: $0, history: Tuning.memoryHistory(after)) }
                    let measured = !(advice?["stale"] is String) ? (advice?["suggested_bytes"] as? NSNumber)?.uint64Value : nil
                    return ended(state, jobs.report?["message"] as? String, measured)
                }
            }, store: { Tuning.storeAutoMeasure(fresh, $0) }, log: log,
            waiting: { Tuning.noteAutoMeasureWaiting(fresh, $0) })
    }

    /// Whether the loaded model's drive (one under /Volumes) is mounted, without reading it (a sleeping drive stays
    /// asleep): seen gone twice in a row makes the status `degraded`, seen back clears it (if the model process
    /// refuses the files it finds there, its own status says so).
    private func checkModelDrive() {
        guard !isUnloaded, let id = scheduler.current?.id, let row = rows().first(where: { $0.id == id }), row.target.hasPrefix("/Volumes/")
        else { return }
        let mounted = Stores.volume(URL(fileURLWithPath: row.target)).mounted
        lock.withLock {
            driveMisses = mounted ? 0 : driveMisses + 1
        }
    }
    /// The `degraded` state's words while the model's drive is gone, else nil.
    private var degraded: String? {
        lock.withLock {
            driveMisses >= 2 ? "The model's drive is not answering: reconnect it; reading resumes by itself, with no reload" : nil
        }
    }
    /// A running model's status with `degraded` (status, words, and its drive's row) when it applies.
    private func markDegraded(_ state: inout [String: Any]) {
        guard let degraded, state["status"] as? String != "unavailable" else { return }
        state["status"] = "degraded"
        state["degraded"] = degraded
        if var from = state["experts_from"] as? [String: Any], var sources = from["sources"] as? [[String: Any]],
            let index = sources.firstIndex(where: { $0["role"] as? String == "all" })
        {
            sources[index]["state"] = "not_connected"
            sources[index]["detail"] = degraded
            from["sources"] = sources
            state["experts_from"] = from
        }
    }

    /// `idle_unload_minutes`: the loaded model unloads after `seconds` with no request and no job, as `origami stop` does
    /// (saved as `stopped`); with `start_on_request` the next request loads it again.
    private func unloadIfIdle(after seconds: TimeInterval) {
        // Idle counts from the last of a request, a start, a load's or a reply's end and a job's end (to a look's interval).
        guard !isUnloaded, scheduler.current != nil, !scheduler.busy, jobs.report?["state"] as? String != "running" else {
            return lock.withLock { activeAt = Date() }
        }
        guard autoMeasure.idleFor >= seconds, Date().timeIntervalSince(lock.withLock { activeAt }) >= seconds else { return }
        log("no request for \(String(format: "%.0f", seconds / 60)) min: unloading the model (idle_unload_minutes)")
        _ = try? unload()
    }

    /// The settings the loaded model runs with (its `--model`), else `saved`.
    private func modelSettings(_ saved: Settings) -> Settings {
        guard let id = scheduler.current?.id, let spec = rows().first(where: { $0.id == id })?.spec else { return saved }
        return (try? Settings.load(flags: flags.merging(["model": [spec]]) { $1 })) ?? saved
    }

    /// `POST v1/jobs`: `{"kind": "add", "source": "<folder|file|owner/name>",
    /// "name": "<name>", "to": "<folder>"}` adds a model (`origami model add`), writing a download or conversion in `to`;
    /// `{"kind": "move", "path": "<item>", "to": "<folder>"}` and `{"kind": "remove", "path": "<item>"}` move or delete an
    /// item of the storage map (`Storage`), nothing else. A move of what the running model reads stops it first and starts
    /// it again once the move is done (or failed), so no model reads a file while it moves.
    /// `then`: the memory search started by `measureIfIdle`, told how it ended.
    private func startJob(_ body: [String: Any], then: (@Sendable (String) -> String?)? = nil) throws {
        DashboardState.mutationLock.lock()
        defer { DashboardState.mutationLock.unlock() }
        // A folder is passed as one `--key=<path>` argument, without a shell: a full path, on one line.
        func folder(_ key: String) throws -> String? {
            guard let text = body[key] as? String, !text.isEmpty else { return nil }
            guard text.hasPrefix("/"), text.count <= 1_024, !text.contains(where: \.isNewline), !text.contains("\0") else {
                throw ServingFailure("\(key) must be a full folder path")
            }
            return text
        }
        switch body["kind"] as? String {
        case "add":
            // One argument, never an option: the source is passed to `model add` as it is, without a shell.
            guard let source = (body["source"] as? String)?.trimmingCharacters(in: .whitespaces), (1...1_024).contains(source.count),
                !source.hasPrefix("-"), !source.contains(where: \.isNewline), !source.contains("\0")
            else {
                throw ServingFailure("source must be a folder, a .gguf file or a Hugging Face repo id (owner/name)")
            }
            let name = (body["name"] as? String).flatMap { $0.isEmpty ? nil : $0 }
            guard name.map({ $0.range(of: "^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$", options: .regularExpression) != nil }) ?? true else {
                throw ServingFailure("A model name is letters, digits, '.', '_' and '-' (up to 64)")
            }
            let to = try folder("to")
            // `--base`: the source is a GGUF of low-bit experts for this exact model, as `origami model add --base` takes it.
            let base = (body["base"] as? String).flatMap { $0.isEmpty ? nil : $0 }
            guard
                base.map({ (1...1_024).contains($0.count) && !$0.hasPrefix("-") && !$0.contains(where: \.isNewline) && !$0.contains("\0") })
                    ?? true
            else { throw ServingFailure("base must be the name of an added model") }
            lock.withLock { foundCache?.date = .distantPast }  // searched again at the next ask; what it found shows meanwhile
            let shown =
                source.hasPrefix("/") || source.hasPrefix("~") ? Models.displayName(URL(fileURLWithPath: Settings.expand(source))) : source
            try jobs.start(
                kind: "add", title: "Adding \(shown)",
                arguments: ["model", "add", source, "--yes"] + (name.map { ["--name=\($0)"] } ?? []) + (to.map { ["--to=\($0)"] } ?? [])
                    + (base.map { ["--base=\($0)"] } ?? []))
        // `origami tune` against this server: Tune, Re-check or Undo.
        case "tune":
            // The reason the dashboard shows next to its tuning buttons, word for word.
            let mode = body["mode"] as? String ?? "tune"
            // The memory search measures whichever model is loaded, with that model's settings (`--model`).
            let saved = try Settings.load(flags: flags)
            let fresh = mode == "memory" ? modelSettings(saved) : saved
            let spec = mode == "memory" ? rows().first(where: { $0.id == scheduler.current?.id })?.spec : nil
            let state = liveState().state
            let block = DashboardState.tuneBlock(state: state, job: jobs.report, reloadRequired: reloadRequired(fresh, state: state))
            guard block.isEmpty else { throw ServingFailure(block, status: 409) }
            if mode == "memory" { lock.withLock { trialEnded = false } }
            guard ["tune", "recheck", "undo", "tradeoffs", "memory"].contains(mode) else {
                throw ServingFailure("mode must be tune, recheck, undo, tradeoffs or memory")
            }
            try jobs.start(
                kind: "tune",
                title: [
                    "tune": "Tuning for this Mac", "recheck": "Re-checking the tuning", "undo": "Undoing the tuning",
                    "tradeoffs": "Comparing quality modes", "memory": "Finding the fastest memory limit",
                ][mode]! + (then == nil ? "" : " while idle"),
                arguments: mode == "memory"
                    ? ["tune", "--port=\(port)", "--memory"] + (spec.map { ["--model=\($0)"] } ?? [])
                    : ["tune", "--port=\(port)", "--review=true"] + ((body["minutes"] as? Int).map { ["--minutes=\($0)"] } ?? [])
                        + (mode == "tradeoffs"
                            ? ["--tradeoffs=true"] + ((body["quality_floor"] as? String).map { ["--quality-floor=\($0)"] } ?? [])
                            : mode == "tune" ? [] : ["--" + mode]),
                then: mode != "memory"
                    ? then
                    : { @Sendable [weak self] state in
                        // However the search ended (its own restore never ran if it was killed): the saved limit is back.
                        // An automatic search's line says so during that reload, and its final words once it is done.
                        if then != nil, let record = Tuning.autoMeasureRecord(fresh) {
                            Tuning.storeAutoMeasure(fresh, record.merging(["line": AutoMeasure.restoringLine(state)]) { $1 })
                        }
                        self?.endMemoryTrial()
                        self?.lock.withLock { self?.trialEnded = false }
                        Tuning.removeTrialHistories()
                        return then?(state)
                    })
        case "recover":
            let fresh = try Settings.load(flags: flags)
            guard scheduler.current == nil else {
                throw ServingFailure("Stop the model before recovering an interrupted move", status: 409)
            }
            guard let id = body["id"] as? String else { throw ServingFailure("Choose a move to recover") }
            try jobs.run(kind: "recover", title: "Recovering move") { [self] _, _ in
                try scheduler.holding(nil) { try MoveJournal.recover(id, settings: fresh) }
            }
        // Remove unusable saved prompts (`SavedPrompts.prune`), with the model loaded: a folder a server holds stays, and one
        // with named or pinned conversations goes only when its path is in `yes`.
        case "prune":
            let fresh = (try? Settings.load(flags: flags)) ?? settings
            try DashboardState.check(body["revision"] as? String, settings: fresh)
            let yes = Set(body["yes"] as? [String] ?? [])
            let unusable = { @Sendable [self] in
                Storage.map(fresh, found: foundForStorage(fresh)).prompts.filter { if case .unusable = $0.state { true } else { false } }
            }
            guard !unusable().isEmpty else { throw ServingFailure("Only an item nothing uses is removed", status: 409) }
            try jobs.run(kind: "prune", title: "Removing unusable saved prompts") { [self] _, _ in
                scheduler.holding(nil) {
                    let builds = unusable()
                    let result = SavedPrompts.prune(
                        builds, all: false, yes: { yes.contains($0.path) }, command: "ticking it")
                    let kept = result.lines.filter { $0.hasPrefix("Keeps ") }.map { $0.dropFirst(6) }
                    return
                        "Removed \(result.removed) of \(builds.count) folders of unusable saved prompts (\(OrigamiPlan.gigabytes(result.freed)))."
                        + (kept.isEmpty ? "" : " Kept " + kept.joined(separator: "; ") + ".")
                }
            }
        case let kind? where ["move", "remove", "delete", "forget", "relink"].contains(kind):
            let fresh = (try? Settings.load(flags: flags)) ?? settings
            // Matched as written, else by the folder it resolves to (/tmp is /private/tmp).
            try DashboardState.check(body["revision"] as? String, settings: fresh)
            let path = body["path"] as? String ?? ""
            let real = Folders.real(path)
            let all = Storage.items(fresh, found: foundForStorage(fresh))
            guard !path.isEmpty,
                let item = all.first(where: { $0.path == path })
                    ?? all.first(where: { $0.readable && real != nil && Folders.real($0.path) == real })
            else {
                throw ServingFailure("Only the items the storage map lists are moved or removed", status: 403)
            }
            // No model whose files these are starts meanwhile (any model for the saved prompts or an item of no known model).
            let holds = item.role == "prompts" || item.model == nil || item.model == "?" ? nil : item.users.union([item.model!])
            if ["delete", "forget", "relink"].contains(kind) {
                let destination = body["to"] as? String ?? ""
                if kind == "delete", item.deleteBlocked != nil { throw ServingFailure(item.deleteBlocked!, status: 409) }
                return try jobs.run(
                    kind: kind, title: (kind == "delete" ? "Deleting " : kind == "forget" ? "Removing " : "Locating ") + item.name
                ) { [self] _, _ in
                    try scheduler.holding(holds) {
                        let active = scheduler.current?.id
                        let affectsActive = Storage.inUse(item, running: active)
                        let work = { () throws -> String in
                            defer {
                                if let reread = try? Settings.load(flags: self.flags) {
                                    self.lock.withLock {
                                        self.loaded = (reread, Self.servedID(reread))
                                        self.foundCache = nil
                                        self.checked = [:]
                                    }
                                }
                            }
                            if kind == "delete" {
                                let result = try Storage.delete(item, settings: fresh)
                                if item.role == "model" {
                                    self.lock.withLock { self.retiredModels.formUnion(item.users.union([item.model ?? ""])) }
                                }
                                return result
                            }
                            if kind == "forget" {
                                try Storage.forget(item, settings: fresh)
                                self.lock.withLock { self.retiredModels.formUnion(item.users.union([item.model ?? ""])) }
                                return "Removed from library; files kept"
                            }
                            return try Storage.relink(item, to: destination, settings: fresh)
                        }
                        if !affectsActive { return try work() }
                        let deletingModel = item.role == "model" && kind != "relink"
                        if deletingModel {
                            try Settings.write(["stopped": true], to: fresh.configFile)
                            self.lock.withLock {
                                self.unloaded = true
                                self.resumeID = nil
                            }
                        }
                        var result: Result<String, Error> = .failure(CLIError("Operation did not complete"))
                        try self.scheduler.reload(deletingModel ? nil : active) { result = Result(catching: work) }
                        return try result.get()
                    }
                }
            }
            if kind == "remove" {
                guard item.unused else {
                    throw ServingFailure(
                        "Only an item nothing uses is removed" + (item.note.map { "; \(item.name): " + $0 } ?? ""), status: 409)
                }
                return try jobs.run(kind: kind, title: "Removing \(item.name)") { [self] _, _ in
                    try scheduler.holding(holds) { try Storage.remove(item, settings: fresh) }
                }
            }
            guard var to = try folder("to") else { throw ServingFailure("to must be a full folder path") }
            // `drive`: `to` names a drive; a pack goes to that drive's store (the Experts card's Move…).
            if body["drive"] as? Bool == true {
                guard item.move == "pack", Stores.volume(URL(fileURLWithPath: to)).mounted else {
                    throw ServingFailure("drive moves a pack to a connected drive")
                }
                let store = HotCopy.store(forDrive: URL(fileURLWithPath: to, isDirectory: true), settings: fresh)
                try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
                to = store.path
            }
            let move = try Storage.plan(item, to: to, settings: fresh, items: all)
            try jobs.run(kind: kind, title: "Moving \(item.name)") { [self] step, cancelled in
                let work = { () throws -> String in
                    defer {
                        if let reread = try? Settings.load(flags: self.flags) {
                            self.lock.withLock { self.loaded = (reread, Self.servedID(reread)) }
                        }
                    }
                    return try Storage.execute(move, settings: fresh, step: step, cancelled: cancelled)
                }
                // While no model runs, none starts until the move is done: a request would read files about to go.
                return try scheduler.holding(holds) {
                    let running = scheduler.current?.id
                    if let running, !Storage.inUse(item, running: running) { return try work() }
                    if let running { step("Stopping \(running) while its files move", nil) }
                    var outcome: Result<String, Error> = .failure(CLIError("\(item.name) was not moved"))
                    do { try scheduler.reload(running) { outcome = Result(catching: work) } } catch {
                        log("after the move: \(ServingFailure.from(error).description)")
                    }
                    return try outcome.get()
                }
            }
        default: throw ServingFailure("kind must be prepare, add, move, remove or prune")
        }
    }

    /// `GET v1/models/found`: models on this Mac Origami could add (`Models.found`), searched again in the background
    /// when two minutes old or on `?fresh` (with the `models_dirs` the page may just have saved). Only a complete search
    /// answers, waited for up to 10 s the first time (20 s afresh), else the last one with `searching` true, which the page
    /// asks again for. A model on a drive the search could not read stays, with that drive's state; one added since leaves.
    private func found(fresh: Bool) -> [String: Any] {
        let start = startSearch(fresh: fresh)
        let deadline = Date().addingTimeInterval(fresh ? 20 : 10)
        while lock.withLock({ searching && (start && fresh || foundCache == nil) }), Date() < deadline { usleep(100_000) }
        let settings = (try? Settings.load(flags: flags)) ?? self.settings
        let known = Set(Self.known(settings).map { $0.lowercased() })
        let (cache, running) = lock.withLock { (foundCache, searching) }
        // Paths compare without case, as the drives Macs format do.
        return [
            "models": (cache?.models ?? []).filter { !known.contains(($0["path"] as? String ?? "").lowercased()) },
            "unreadable": cache?.unreadable ?? [],
            "folders": settings.modelsDirs.map(\.path), "searching": running,
        ]
    }

    /// Starts a model search when one is due (two minutes old, or `fresh`); one asked afresh during a search (which may
    /// predate a folder just saved) follows it.
    @discardableResult private func startSearch(fresh: Bool) -> Bool {
        let start = lock.withLock { () -> Bool in
            if searching {
                searchAgain = searchAgain || fresh
                return false
            }
            guard fresh || foundCache.map({ Date().timeIntervalSince($0.date) >= 120 }) ?? true else { return false }
            searching = true
            return true
        }
        if start { Thread.detachNewThread { [self] in search() } }
        return start
    }

    /// The storage map's models found on this Mac: the last complete search, never waited for (a search is started when
    /// due). It is incomplete before the first search ends, when a drive did not answer it, and when a model it left
    /// out as registered or configured is no longer (searched again).
    private func foundForStorage(_ settings: Settings) -> Storage.Found {
        startSearch(fresh: false)
        guard let cache = lock.withLock({ foundCache }) else { return ([], "the search for models on this Mac has not finished") }
        if let partial = cache.partial { return (cache.models, partial) }
        guard cache.known.isSubset(of: Self.known(settings)) else {
            startSearch(fresh: true)
            return (cache.models, "the search for models on this Mac has not finished")
        }
        return (cache.models, nil)
    }

    /// The paths registered or configured, which the search leaves out.
    private static func known(_ settings: Settings) -> [String] {
        ((try? ModelRegistry.entries(settings)) ?? []).compactMap { $0["path"] as? String }
            + [settings.model.map(Settings.expand)].compactMap { $0 }
    }

    /// One model search; the drives it could not read keep the models a search found there before.
    private func search() {
        repeat {
            let settings = (try? Settings.load(flags: flags)) ?? self.settings
            let known = Set(Self.known(settings))
            let (models, unreadable, skipped, partial) = Models.found(settings: settings, known: known)
            lock.withLock {
                let kept = (foundCache?.models ?? []).compactMap { model -> [String: Any]? in
                    guard let drive = model["drive"] as? String, let state = skipped[drive] else { return nil }
                    return model.merging(["state": state.rawValue]) { $1 }
                }
                foundCache = (
                    Date(), models.map { $0.merging(["state": Stores.Availability.available.rawValue]) { $1 } } + kept, unreadable,
                    skipped.isEmpty
                        ? partial : skipped.keys.sorted().joined(separator: ", ") + " did not answer the search for models on this Mac",
                    known
                )
                searching = searchAgain
                searchAgain = false
            }
        } while lock.withLock({ searching })
    }

    /// A listed model's availability: the last check's, a check is started when there is none (or, for the dashboard's
    /// `recheck`, when it is 15 s old), and a check that has not answered says `checking`, then `slow` after 3 s, and so
    /// does the last state once it has waited 30 s. One check per model runs at a time: a drive waiting on a permission
    /// prompt holds one thread, not one per ask (a new one only after 30 s, for the dashboard).
    private func availability(_ row: Row, recheck: Bool) -> Stores.Availability {
        let target = row.target
        let now = Date()
        let (last, since) = lock.withLock { () -> ((state: Stores.Availability, bytes: UInt64?, date: Date)?, Date?) in
            let last = checked[target]
            // A check that has waited 30 s is abandoned when the dashboard asks again (a drive allowed since may answer
            // a new one), its model kept `slow` meanwhile, so the label does not fall back to `checking`.
            let abandoned = recheck && checking[target].map { now.timeIntervalSince($0) > 30 } ?? false
            if abandoned { checked[target] = (.slow, last?.bytes, now) }
            if abandoned || checking[target] == nil && (last == nil || recheck && now.timeIntervalSince(last!.date) > 15) {
                checking[target] = now
                let settings = loaded.settings
                Thread.detachNewThread { [self] in
                    let path =
                        target.hasPrefix("locate:")
                        ? Models.locate(target == "locate:" ? nil : String(target.dropFirst(7)), settings: settings)?.path : target
                    var state = path.map(Stores.probe) ?? .missing
                    // A model is a .gguf file, or a folder with a config.json or a GGUF.
                    if state == .available, let path, !path.lowercased().hasSuffix(".gguf"),
                        !FileManager.default.fileExists(atPath: path + "/config.json"), !Models.isGGUF(path)
                    {
                        state = .missing
                    }
                    // A split GGUF counts every part (its headers name them).
                    let bytes = path.flatMap { path -> UInt64? in
                        guard path.lowercased().hasSuffix(".gguf"), state == .available else { return nil }
                        let url = URL(fileURLWithPath: path)
                        if let gguf = try? GGUFSource(url), gguf.files.count > 1 {
                            return gguf.files.reduce(0) {
                                $0 + UInt64(max(0, Models.fileSize(gguf.folder.appendingPathComponent($1)) ?? 0))
                            }
                        }
                        return Models.fileSize(url).map { UInt64(max(0, $0)) }
                    }
                    let recommended = state == .available ? path.map(Models.recommendedSampling) : nil
                    lock.withLock {
                        checked[target] = (state, bytes ?? checked[target]?.bytes, Date())
                        checking[target] = nil
                        if let recommended { recommendations[target] = recommended }
                    }
                }
            }
            return (checked[target], checking[target])
        }
        let waited = since.map { now.timeIntervalSince($0) } ?? 0
        if let last, waited < 30 { return last.state }
        return since == nil ? .checking : waited > 3 ? .slow : .checking
    }

    /// `GET v1/models/check?source=`: what adding a folder, a .gguf file or a Hugging Face repo would do, from its headers
    /// alone: the family, `runs` (as it is), `converts` (once, into the 4-bit layout or a GLM GGUF) or `refused` with the reason,
    /// and whether the drive it is written to (`to`, else the data folder's) has the room a download or conversion needs: its
    /// size and 10 GB to spare, the conversion's counted as `model add` counts it (`Models.conversionBytes`), a Hugging Face
    /// checkpoint's from its headers before it is downloaded, beside the download.
    private func check(_ source: String, to: String?) -> [String: Any] {
        let settings = self.settings
        let destination = to.map { URL(fileURLWithPath: Settings.expand($0), isDirectory: true) } ?? settings.modelsDir
        guard let drive = Folders.space(destination.path) else {
            return [
                "source": source, "verdict": "refused", "reason": Folders.permission(destination.path),
                "permission": Stores.driveName(destination.path),
            ]
        }
        guard drive.mounted else {
            return ["source": source, "verdict": "refused", "reason": "\(Stores.driveName(destination.path)) is not connected"]
        }
        var out: [String: Any] = ["source": source, "drive": drive.name, "free_bytes": drive.free, "destination": destination.path]
        func refused(_ reason: String) -> [String: Any] { out.merging(["verdict": "refused", "reason": reason]) { $1 } }
        let report: CompatibilityReport
        let needs: UInt64
        do {
            if source.hasPrefix("/") || source.hasPrefix("~") {
                let url = URL(fileURLWithPath: Settings.expand(source))
                guard Stores.answers(url, within: 5) else {
                    out["permission"] = Stores.driveName(url.path)
                    return refused("macOS has not let Origami read \(Stores.driveName(url.path)) yet")
                }
                guard FileManager.default.fileExists(atPath: url.path) else { return refused("Nothing at \(url.path)") }
                report = try ModelCompatibility.check(local: url)
                // What `model add` writes converting it (a GLM GGUF, Qwen's 4-bit layout), as it counts it; a model that runs is read where it is.
                if report.verdict == .converts, let family = report.family {
                    needs = Models.conversionNeeds(try Models.conversionBytes(url, family: family, settings: settings))
                } else {
                    needs = 0
                }
            } else if source.range(of: "^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$", options: .regularExpression) != nil {
                let reports = try ModelCompatibility.check(hub: HubRepo(source))
                guard let first = reports.first else { return refused("\(source) holds no safetensors or GGUF model") }
                guard reports.count == 1, !first.format.lowercased().contains("gguf") else {
                    return refused("\(source) holds GGUF files: download the one you want, then add it from its folder")
                }
                // A checkpoint that converts is downloaded beside where it converts to: both, and 10 GB to spare.
                let converts =
                    first.verdict == .converts
                    ? try first.family.map { family in
                        try Models.conversionBytes(
                            hub: HubRepo(source), base: String(first.source.dropFirst(source.count + 1)), family: family, settings: settings
                        )
                    } ?? 0 : 0
                report = first
                // A download stopped part way resumes: what is already in its folder is not needed again, as `model add`
                // counts it (its own files and their `.part`s).
                let folder = destination.appendingPathComponent(source.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
                let present =
                    (FileManager.default.enumerator(at: folder, includingPropertiesForKeys: [.fileSizeKey])?.compactMap {
                        (try? ($0 as? URL)?.resourceValues(forKeys: [.fileSizeKey]))?.fileSize
                    }.reduce(0, +)).map { UInt64(min($0, Int(max(0, first.totalBytes)))) } ?? 0
                needs = Models.conversionNeeds(UInt64(max(0, first.totalBytes)) - present + converts)
                if present > 0 { out["present_bytes"] = present }
            } else {
                return refused("Give a folder, a .gguf file or a Hugging Face repo id (owner/name)")
            }
        } catch { return refused("\(error)") }
        guard report.family != nil, report.verdict != .refused else { return refused(report.reason) }
        return out.merging([
            "name": source.contains("/") && !source.hasPrefix("/") && !source.hasPrefix("~")
                ? source : Models.displayName(URL(fileURLWithPath: Settings.expand(source))),
            "family": report.family!.displayName, "bytes": report.totalBytes, "verdict": report.verdict.rawValue,
            "reason": report.reason, "needs_bytes": needs, "fits": drive.free >= needs,
        ]) { $1 }
    }

    /// Whether every model is listed and chosen by name (`list_models`), or only `origami`.
    private var listsAll: Bool { settings.listModels }

    /// Whether a listed model reads images, without opening it.
    /// shortcut: by family (Qwen's checkpoints keep their vision tower); read the checkpoint's tensor names if a text-only
    /// Qwen checkpoint is ever listed. The running model's own server answers from its checkpoint.
    private static func vision(_ row: Row) -> Bool { row.family == .qwen }

    /// `/v1/models`, `/api/tags` or `/api/version`: with `all`, every model and, where non-exact levels are allowed, one
    /// `<id>-<level>` per level it offers, as a one-model server lists its own; otherwise `origami` alone, the configured model.
    /// `recheck`: the dashboard asks, so each model's availability is checked again when 15 s old.
    private func listing(_ path: String, rows: [Row], all: Bool, recheck: Bool = false) -> [String: Any] {
        if path == "/api/version" { return ["version": version] }
        let running = scheduler.current?.id
        // Each model's own fast levels.
        func levels(_ row: Row) -> [String] { settings.allowFast ? Array(QualityDial.offered(row.family).dropFirst()) : [] }
        let named =
            all
            ? rows.flatMap { row in [(row, row.id)] + levels(row).map { (row, row.id + "-" + $0) } }
            : rows.prefix(1).map { ($0, Self.setupID) }
        // Only the models listed are checked (which also learns a GGUF's size): a client's plain listing touches no other drive.
        let states = Dictionary((all ? rows : Array(rows.prefix(1))).map { ($0.id, availability($0, recheck: recheck)) }) { first, _ in
            first
        }
        if path == "/api/tags" {
            // `ollama list` prints the first 12 characters of the digest (and crashes on a shorter one).
            func digest(_ name: String) -> String { SHA256.hash(data: Data(name.utf8)).map { String(format: "%02x", $0) }.joined() }
            return [
                "models": named.map { row, name in
                    let (details, abilities) = LocalHTTPServer.ollamaDescription(
                        row.family, quantization: row.quantization, vision: Self.vision(row))
                    return [
                        "name": name, "model": name, "modified_at": row.added, "size": row.bytes, "digest": digest(name),
                        "details": details,
                        "capabilities": abilities,
                    ] as [String: Any]
                }
            ]
        }
        // The context is the setting every model starts with; its own server may serve less on a Mac with little memory.
        let context = settings.context
        // The sampling each model's server starts with: its settings over its recommendation, once its check has read that.
        let sampling = Dictionary(
            (all ? rows : Array(rows.prefix(1))).compactMap { row -> (String, [String: Any])? in
                guard let recommended = lock.withLock({ recommendations[row.target] }) else { return nil }
                let own = row.setting.flatMap { try? Settings.load(flags: flags.merging(["model": [$0]]) { $1 }) } ?? settings
                return (
                    row.id, ["defaults": own.sampling(recommended).wire, "recommended": recommended.wire, "override": own.samplingOverride]
                )
            }
        ) { first, _ in first }
        return [
            "object": "list",
            "data": named.map { row, name in
                var origami: [String: Any] = [
                    "model": row.id, "family": row.family.rawValue, "running": row.id == running, "default": row.id == rows.first?.id,
                    "bytes": row.bytes, "quantization": row.quantization, "setting": row.setting ?? "",
                    "availability": (row.id == running ? .available : states[row.id] ?? .checking).rawValue,
                    "drive": row.target.hasPrefix("/") ? Stores.driveName(row.target) : "internal drive",
                ]
                origami["sampling"] = sampling[row.id]
                // A low-bit expert variant names its exact base: its replies differ from that model's.
                if let base = row.variantOf { origami["variant_of"] = base }
                return LocalHTTPServer.modelEntry(name, family: row.family, vision: Self.vision(row), context: context, origami: origami)
            },
        ]
    }

    // MARK: sockets

    /// Sends the request to the child and its response back until the child closes (it answers one request per
    /// connection) through `BoundedHTTPRelay` (1 MiB waiting at most, so the child never waits on a client that reads). A
    /// client that leaves, sends more, or takes no byte for 30 s while bytes wait closes the child's connection, which
    /// cancels the request there (the conversation is kept), as does `cancelled` (a stop that waited long enough), the
    /// client then getting `stoppedFailure` if nothing came yet.
    /// `streaming`: the front already sent this stream's headers (`StreamWait`), so the model's status line and headers
    /// are dropped, and a refusal it answers instead of a stream goes out as a stream error event in that format.
    /// `raw` (as `BoundedHTTPIngress.forwarded` made it) with its JSON body's `origami_time_limit` lowered by `waited`,
    /// or `raw` itself when it sets none or nothing was waited; nil when the limit has passed.
    static func limitFromArrival(_ raw: Data, body: Data, waited: TimeInterval) -> Data? {
        guard waited >= 0.05, body.range(of: Data("origami_time_limit".utf8)) != nil,
            var object = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any],
            let limit = (object["origami_time_limit"] as? NSNumber)?.doubleValue,
            let end = raw.range(of: Data("\r\n\r\n".utf8)), let text = String(data: raw[..<end.lowerBound], encoding: .isoLatin1)
        else { return raw }
        guard limit > waited else { return nil }
        object["origami_time_limit"] = limit - waited
        guard let sent = try? JSONSerialization.data(withJSONObject: object) else { return raw }
        let head = text.split(separator: "\r\n", omittingEmptySubsequences: false)
            .map { $0.lowercased().hasPrefix("content-length:") ? "content-length: \(sent.count)" : String($0) }
        return (head.joined(separator: "\r\n") + "\r\n\r\n").data(using: .isoLatin1)! + sent
    }

    private static func pipe(
        _ request: Data, client: Int32, port: UInt16, cancelled: () -> Bool, lifecycle: HTTPRequestLifecycle, streaming: WireFormat? = nil,
        endOfInputLeaves: Bool = true
    ) {
        lifecycle.mark("service_start")
        lifecycle.fields(["outcome": "upstream_unavailable"])
        defer { lifecycle.mark("service_end") }
        func localFailure(_ failure: ServingFailure) {
            lifecycle.fields(["http_status": failure.status, "error_code": failure.code])
            if let streaming { return streamError(client, failure, format: streaming, lifecycle: lifecycle) }
            reply(client, failure)
        }
        let upstream = connect(port: port)
        guard upstream >= 0 else {
            return localFailure(ServingFailure("The model's server is not answering", status: 502, type: "engine_unavailable"))
        }
        defer { close(upstream) }
        guard send(upstream, request) else {
            return localFailure(ServingFailure("The model's server is not answering", status: 502, type: "engine_unavailable"))
        }
        if streaming != nil {
            // The model's head: kept back; its body follows the front's own headers.
            var head = Data()
            var chunk = [UInt8](repeating: 0, count: 16 * 1_024)
            while head.range(of: Data("\r\n\r\n".utf8)) == nil {
                var ready = pollfd(fd: upstream, events: Int16(POLLIN), revents: 0)
                if poll(&ready, 1, 500) == 0 {
                    if cancelled() { return localFailure(stoppedFailure) }
                    if left(client, endOfInputLeaves: endOfInputLeaves) { return lifecycle.fields(["outcome": "disconnected_queue"]) }
                    continue
                }
                let count = read(upstream, &chunk, chunk.count)
                guard count > 0 else {
                    return localFailure(ServingFailure("The model's server is not answering", status: 502, type: "engine_unavailable"))
                }
                head.append(contentsOf: chunk[0..<count])
            }
            let end = head.range(of: Data("\r\n\r\n".utf8))!.upperBound
            var body = Data(head[end...])
            let status = Int(String(decoding: head.prefix(12), as: UTF8.self).split(separator: " ").dropFirst().first ?? "") ?? 502
            guard status == 200 else {
                // A refusal (JSON, Connection: close): read to its end, then said in the stream's format.
                while case let count = read(upstream, &chunk, chunk.count), count > 0 { body.append(contentsOf: chunk[0..<count]) }
                let object = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any]
                let error = object?["error"] as? [String: Any]
                return localFailure(
                    ServingFailure(
                        error?["message"] as? String ?? "The model's server answered \(status)", status: status,
                        type: error?["type"] as? String ?? "engine_unavailable", code: error?["code"] as? String))
            }
            lifecycle.fields(["outcome": "proxied_response"])
            guard send(client, body) else { return lifecycle.fields(["outcome": "client_left"]) }
        }
        var forwarded = streaming != nil
        let outcome = BoundedHTTPRelay.relay(upstream: upstream, client: client, endOfInputLeaves: endOfInputLeaves, cancelled: cancelled) {
            forwarded = true
            lifecycle.fields(["outcome": "proxied_response"])
        }
        if outcome == .cancelled {
            lifecycle.fields(["outcome": "cancelled_proxy"])
            if !forwarded { localFailure(stoppedFailure) }
        } else if outcome == .clientStalled {
            lifecycle.fields(["outcome": "client_stalled"])
        } else if outcome == .finished && !forwarded {
            // It closed without a byte: the model's process ended under this request (the log names its signal).
            localFailure(
                ServingFailure(
                    "The model's process stopped while answering; see the server log. The next request starts it again",
                    status: 502, type: "engine_unavailable"))
        }
    }

    /// A stream error event in `format` (after the headers `StreamWait` sent), as the model's own server ends a stream.
    private static func streamError(_ client: Int32, _ failure: ServingFailure, format: WireFormat, lifecycle: HTTPRequestLifecycle) {
        lifecycle.fields(["error_code": failure.code, "outcome": "stream_error"])
        let id = (format == .anthropic ? "msg_" : format == .responses ? "resp_" : "chatcmpl-") + UUID().uuidString
        let wire = WireWriter(format: format, id: id, created: Int(Date().timeIntervalSince1970), model: "", includeUsage: false)
        for frame in (try? wire.errorFrame(failure)) ?? [] { guard send(client, frame) else { return } }
    }

    /// A streamed request waiting for its model (`pass`): after 10 s it gets the stream's headers and then a keepalive
    /// every 10 s until `finish`, which says whether the headers went out (nothing is written after it returns).
    final class StreamWait: @unchecked Sendable {
        private let lock = NSLock()
        private let done = DispatchSemaphore(value: 0)
        private var finished = false, started = false, failed = false
        init(_ client: Int32) {
            Thread.detachNewThread { [self] in
                while done.wait(timeout: .now() + 10) == .timedOut {
                    let alive = lock.withLock {
                        guard !finished else { return false }
                        if !started {
                            let head = [
                                "HTTP/1.1 200 OK", "Content-Type: text/event-stream; charset=utf-8", "Connection: close",
                                "Cache-Control: no-store", "X-Content-Type-Options: nosniff", "X-Origami-Version: \(version)",
                            ]
                            started = true
                            guard ModelFront.send(client, Data((head.joined(separator: "\r\n") + "\r\n\r\n").utf8)) else {
                                failed = true
                                return false
                            }
                        }
                        failed = !ModelFront.send(client, ServingWire.keepAlive)
                        return !failed
                    }
                    if !alive { return }
                }
            }
        }
        /// Once the headers went out, whether a write to the client failed (it left); nil before.
        var lost: Bool? { lock.withLock { started ? failed : nil } }
        /// Stops the keepalives; true when the headers went out.
        func finish() -> Bool {
            let sent = lock.withLock {
                finished = true
                return started
            }
            done.signal()
            return sent
        }
    }

    /// Whether a waiting client has left (or sent bytes past its request).
    private static func left(_ client: Int32, endOfInputLeaves: Bool = true) -> Bool {
        BoundedHTTPRelay.left(client, endOfInputLeaves: endOfInputLeaves)
    }

    private static func send(_ fd: Int32, _ data: Data) -> Bool {
        data.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { return true }
            var offset = 0
            while offset < bytes.count {
                let sent = write(fd, base + offset, bytes.count - offset)
                if sent < 0 && errno == EINTR { continue }
                guard sent > 0 else { return false }
                offset += sent
            }
            return true
        }
    }

    /// A stop does not end by itself: `X-Should-Retry: false` keeps the OpenAI SDKs from retrying it.
    private static func reply(_ fd: Int32, _ failure: ServingFailure) {
        reply(fd, status: failure.status, ["error": failure.wire], extra: failure.code == "stopped" ? ["X-Should-Retry: false"] : [])
    }
    /// The dashboard page or one of its files, with the one-model server's headers: it loads nothing from elsewhere.
    private static func page(_ fd: Int32, head: Bool, asset: (type: String, body: Data)?) {
        let (type, body) = asset ?? ("text/html; charset=utf-8", Data(DashboardPage.html.utf8))
        let lines = [
            "HTTP/1.1 200 OK", "Content-Type: \(type)", "Content-Length: \(body.count)", "Connection: close",
            "Cache-Control: \(asset == nil ? "no-store" : "max-age=86400")", "X-Content-Type-Options: nosniff",
            "X-Origami-Version: \(version)",
            "Referrer-Policy: no-referrer", "Content-Security-Policy: \(DashboardPage.csp)",
        ]
        _ = send(fd, Data((lines.joined(separator: "\r\n") + "\r\n\r\n").utf8) + (head ? Data() : body))
    }
    private static func reply(_ fd: Int32, status: Int, _ object: [String: Any], head: Bool = false, extra: [String] = []) {
        // ServingWire.json refuses a value JSON cannot hold (a NaN) instead of aborting the front.
        let body = (try? ServingWire.json(object)) ?? Data("{}".utf8)
        let phrases = [
            200: "OK", 202: "Accepted", 400: "Bad Request", 403: "Forbidden", 409: "Conflict", 401: "Unauthorized", 404: "Not Found",
            405: "Method Not Allowed", 408: "Request Timeout", 411: "Length Required", 429: "Too Many Requests",
            413: "Content Too Large", 417: "Expectation Failed", 431: "Request Header Fields Too Large", 502: "Bad Gateway",
            503: "Service Unavailable",
        ]
        var lines = [
            "HTTP/1.1 \(status) \(phrases[status] ?? "Internal Server Error")", "Content-Type: application/json",
            "Content-Length: \(body.count)",
            "Connection: close", "Cache-Control: no-store", "X-Content-Type-Options: nosniff", "X-Origami-Version: \(version)",
        ]
        if status == 401 { lines.append("WWW-Authenticate: Bearer") }
        lines += extra
        _ = send(fd, Data((lines.joined(separator: "\r\n") + "\r\n\r\n").utf8) + (head ? Data() : body))
    }

    /// No SIGPIPE on a closed peer; a client gets 30 s per read and write, as the one-model server allows.
    private static func configure(_ fd: Int32) {
        var one: Int32 = 1
        var timeout = timeval(tv_sec: 30, tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
    }

    private static func loopback(_ port: UInt16) -> sockaddr_in {
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = port.bigEndian
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        return address
    }
    private static func withAddress(
        _ address: inout sockaddr_in, _ body: (UnsafeMutablePointer<sockaddr>, UnsafeMutablePointer<socklen_t>) -> Int32
    ) -> Int32 {
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        return withUnsafeMutablePointer(to: &address) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { body($0, &length) } }
    }

    /// A connected socket to 127.0.0.1:`port` (without SIGPIPE), or -1.
    private static func connect(port: UInt16) -> Int32 {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return -1 }
        var address = loopback(port)
        guard withAddress(&address, { Darwin.connect(fd, $0, $1.pointee) }) == 0 else {
            close(fd)
            return -1
        }
        var one: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
        return fd
    }

    /// A port no one listens on now, for a child (the kernel picks it).
    private static func freePort() throws -> UInt16 {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw ServingFailure("No socket for a model's server", status: 503, type: "engine_unavailable") }
        defer { close(fd) }
        var address = loopback(0)
        guard withAddress(&address, { bind(fd, $0, $1.pointee) }) == 0, withAddress(&address, { getsockname(fd, $0, $1) }) == 0 else {
            throw ServingFailure("No free port for a model's server", status: 503, type: "engine_unavailable")
        }
        return UInt16(bigEndian: address.sin_port)
    }

    /// The peer's address when the connection comes from another device, nil when from this Mac's loopback address
    /// (its network address counts as another device, which could also use it).
    private static func peer(_ fd: Int32) -> String? {
        var address = sockaddr_in()
        guard withAddress(&address, { getpeername(fd, $0, $1) }) == 0, address.sin_family == sa_family_t(AF_INET) else { return "unknown" }
        guard UInt32(bigEndian: address.sin_addr.s_addr) >> 24 != 127 else { return nil }
        var text = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
        return inet_ntop(AF_INET, &address.sin_addr, &text, socklen_t(text.count)) != nil
            ? String(decoding: text.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self) : "unknown"
    }

    /// The front's listening socket on 127.0.0.1:`port`, or with `lan` on every IPv4 interface, and the port it got.
    static func listen(port: Int, lan: Bool = false) throws -> (fd: Int32, port: UInt16) {
        let where_ = lan ? "0.0.0.0" : "127.0.0.1"
        // SO_REUSEADDR lets the wildcard bind beside a server on 127.0.0.1:`port`, which would keep this Mac's requests.
        if lan && port != 0 { close(try listen(port: port).fd) }
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0, (0...65_535).contains(port) else { throw CLIError("Cannot listen on \(where_):\(port)") }
        var one: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, socklen_t(MemoryLayout<Int32>.size))
        var address = loopback(UInt16(port))
        if lan { address.sin_addr.s_addr = INADDR_ANY }
        guard withAddress(&address, { bind(fd, $0, $1.pointee) }) == 0, Darwin.listen(fd, 64) == 0,
            withAddress(&address, { getsockname(fd, $0, $1) }) == 0
        else {
            let (code, reason) = (errno, String(cString: strerror(errno)))
            close(fd)
            if code == EADDRINUSE {
                throw CLIError(
                    "Port \(port) is already in use, perhaps by another Origami server (`origami status --port \(port)` shows it).\n"
                        + "Stop that one, or use another port: origami serve --port \(port == 65_535 ? port - 1 : port + 1)")
            }
            throw CLIError("Cannot listen on \(where_):\(port) (\(reason)). Use another port: origami serve --port <n>")
        }
        return (fd, UInt16(bigEndian: address.sin_port))
    }

    // MARK: serve

    static func serve(settings: Settings, flags: [String: [String]]) async throws {
        let log: @Sendable (String) -> Void = { line in
            FileHandle.standardError.write(Data("\(ISO8601DateFormatter().string(from: Date())) origami: \(line)\n".utf8))
        }
        var settings = settings
        // `listen lan` set by hand, without a key for other devices: one is made, as `origami config set listen lan` does.
        let key = settings.lanKeyFor([:])
        if !key.isEmpty {
            try Settings.write(key, to: settings.configFile)
            settings = try Settings.load(flags: flags)
        }
        let (listener, port) = try listen(port: settings.port, lan: settings.listenLAN)
        let front = ModelFront(settings: settings, flags: flags, port: Int(port), log: log)
        let ids = front.rows().map(\.id)
        log(
            settings.listModels
                ? "listening on http://127.0.0.1:\(port), \(ids.count == 1 ? "model" : "models") \(ids.joined(separator: ", ")); starting \(front.firstID)"
                : "listening on http://127.0.0.1:\(port) as `\(setupID)`: \(front.firstID)")
        if settings.listenLAN {
            log(
                "also listening on the network (listen lan): " + Access.urls(port: Int(port)).joined(separator: ", ")
                    + (settings.remoteKey == nil
                        ? "; other devices need no key (lan_key off)"
                        : "; other devices need the key (\(settings.apiKey != nil ? "api_key" : "lan_key")), which travels in clear "
                            + "over plain HTTP: for TLS, see docs/integration.md#tls"))
        }
        // The configured model starts at once. If it cannot, the server stays up and says why (`/health`, the dashboard),
        // so the page can fix it; the next request or reload tries again. Stopped (`stopped`), none starts until Start.
        if settings.stopped {
            log("stopped (origami stop, or Stop and unload in the dashboard): no model loaded until origami start or the dashboard's Start")
        } else {
            Thread.detachNewThread {
                do {
                    _ = try front.scheduler.acquire(nil, fallback: front.firstID)
                    front.scheduler.release()
                } catch {}
            }
        }
        Thread.detachNewThread { front.run(listener: listener) }
        let previousINT = signal(SIGINT, SIG_IGN)
        let previousTERM = signal(SIGTERM, SIG_IGN)
        let interrupt = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
        let terminate = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global())
        // Ctrl-C reaches the child too (one process group), which stops as a one-model server does; SIGTERM is passed on.
        interrupt.setEventHandler { front.stop() }
        terminate.setEventHandler { front.stop() }
        interrupt.resume()
        terminate.resume()
        defer {
            interrupt.cancel()
            terminate.cancel()
            signal(SIGINT, previousINT)
            signal(SIGTERM, previousTERM)
        }
        while !front.isStopped { try await Task.sleep(for: .milliseconds(200)) }
        log("stopping")
        front.scheduler.stopAll()
        if !front.stopAndJoin() { log("request cleanup or metrics persistence did not finish within 30 s") }
        close(listener)
    }
}
