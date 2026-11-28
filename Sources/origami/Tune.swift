import Darwin
import Foundation
import OrigamiEngine

/// `origami tune` (and the dashboard's tuning, run as a job): finds this Mac's fastest memory limit for the running model
/// (`TuneMemory.swift`), or with `--tradeoffs` times each quality level (`TuningTradeoffs.swift`). Results are kept in
/// `<data_dir>/tuning.json`, per model and context, bound to this Mac, its drives and `SelfTuning.version`. Entries an
/// older build kept for its timing-switch comparisons (a model's key alone, `partial@…`) are left as they are and ignored.
enum Tuning {
    static func file(_ settings: Settings) -> URL { settings.dataDir.appendingPathComponent("tuning.json") }
    static func proposalFile(_ settings: Settings) -> URL { settings.dataDir.appendingPathComponent("tuning-proposal.json") }
    static func historyFile(_ settings: Settings) -> URL { settings.dataDir.appendingPathComponent("tuning-history.jsonl") }
    static func memoryHistory(_ settings: Settings) -> MemoryHistory {
        // A model loaded at a trial limit (`origami tune --memory`) neither reads nor teaches this Mac's history: its cuts
        // are the measurement, kept in a throwaway file.
        let trialHistory = ProcessInfo.processInfo.environment[trialVariable].map { _ in
            FileManager.default.temporaryDirectory.appendingPathComponent("origami-memory-trial-\(getpid()).json")
        }
        if let trialHistory { try? FileManager.default.removeItem(at: trialHistory) }
        return MemoryHistory(
            file: trialHistory ?? settings.dataDir.appendingPathComponent("memory-history.json"),
            identity: "\(macModel) \(ProcessInfo.processInfo.physicalMemory)",
            scope: MemoryHistory.Scope(model: settings.model ?? "", context: settings.context))
    }

    /// Removes the throwaway histories of trial models that have ended (`memoryHistory`); a running one keeps its own.
    static func removeTrialHistories() {
        let folder = FileManager.default.temporaryDirectory
        for name in (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? [] {
            guard name.hasPrefix("origami-memory-trial-"), name.hasSuffix(".json"),
                let pid = pid_t(name.dropFirst("origami-memory-trial-".count).dropLast(".json".count)),
                kill(pid, 0) != 0 && errno == ESRCH
            else { continue }
            try? FileManager.default.removeItem(at: folder.appendingPathComponent(name))
        }
    }

    static var macModel: String {
        var size = 0
        guard sysctlbyname("hw.model", nil, &size, nil, 0) == 0, size > 0 else { return "unknown" }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname("hw.model", &bytes, &size, nil, 0) == 0 else { return "unknown" }
        return String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// What a result is bound to: the Mac, its GPU and memory, the tuning version, the model and the drives it reads.
    static func identity(model: String, gpu: String, drives: [URL]) -> [String: Any] {
        let uuids = Set(drives.compactMap { try? $0.resourceValues(forKeys: [.volumeUUIDStringKey]).volumeUUIDString }).sorted()
        return [
            "tuning_version": SelfTuning.version, "mac": macModel, "gpu": gpu, "memory_bytes": ProcessInfo.processInfo.physicalMemory,
            "model": String(model.prefix(12)), "drives": uuids.joined(separator: ","),
            "macos": ProcessInfo.processInfo.operatingSystemVersion.majorVersion,
        ]
    }

    /// A serving model's tuning key and identity, for /v1/status: what a measurement is kept under and bound to.
    struct State: @unchecked Sendable {
        let key: String, identity: [String: Any]
        init(model: String, context: Int, gpu: String, drives: [URL]) {
            key = Tuning.key(model: model, context: context)
            identity = Tuning.identity(model: model, gpu: gpu, drives: drives)
        }
        var report: [String: Any] { ["version": SelfTuning.version, "key": key, "identity": identity] }
    }

    static func entries(_ settings: Settings) throws -> [String: Any]? {
        guard let data = try? Data(contentsOf: file(settings)) else { return nil }
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any], let entries = object["entries"] as? [String: Any]
        else {
            throw CLIError("tuning.json holds no entries")
        }
        return entries
    }

    /// A result's key in tuning.json: the model's identity (truncated) and the context.
    static func key(model: String, context: Int) -> String { "\(model.prefix(12))@\(context)" }

    /// Writes (or with nil removes) one entry, keeping the others.
    static func save(_ settings: Settings, key: String, entry: [String: Any]?) throws {
        try StateFile.locked(file(settings)) {
            // An unreadable tuning.json is set aside, never rewritten from empty (`StateFile`).
            var all: [String: Any]
            do { all = try entries(settings) ?? [:] } catch {
                guard StateFile.setAside(file(settings)) else {
                    throw CLIError("\(file(settings).path) cannot be read and was left as it is; nothing was saved")
                }
                all = [:]
            }
            all[key] = entry
            try JSONSerialization.data(withJSONObject: ["entries": all], options: [.prettyPrinted, .sortedKeys]).write(
                to: file(settings), options: .atomic)
        }
    }

    /// `origami report`: the saved results and the last run.
    static func reportFields(_ settings: Settings) -> [String: Any] {
        let last = (try? String(contentsOf: historyFile(settings), encoding: .utf8))?.split(separator: "\n").last
            .flatMap { try? JSONSerialization.jsonObject(with: Data($0.utf8)) }
        return ["tuning": ((try? entries(settings)) ?? nil).map { $0 as Any } ?? NSNull(), "tuning_last_run": last ?? NSNull()]
    }

    // MARK: origami tune

    final class Aborted: Error, CustomStringConvertible {
        let description: String
        init(_ text: String) { description = text }
    }

    /// The front's model did not start (`/v1/status` says `failed`), with its reason.
    struct NotStarted: Error, CustomStringConvertible { let description: String }

    struct Client {
        let base: URL, key: String?
        var deadline: Date? = nil
        func call(_ path: String, body: [String: Any]? = nil) async throws -> [String: Any] {
            var request = URLRequest(url: base.appendingPathComponent(path))
            request.timeoutInterval = min(30, max(0.01, deadline?.timeIntervalSinceNow ?? 30))
            if let deadline, Date() >= deadline { throw Aborted("Time limit reached; no settings changed") }
            if let key { request.setValue("Bearer " + key, forHTTPHeaderField: "Authorization") }
            if let body {
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONSerialization.data(withJSONObject: body)
            }
            let (data, response): (Data, URLResponse)
            do { (data, response) = try await URLSession.shared.data(for: request) } catch {
                // Cut short at the limit (its timeout is the time left): the limit is the reason.
                if let deadline, Date() >= deadline { throw Aborted("Time limit reached; no settings changed") }
                throw Aborted("Origami stopped answering (it may be restarting)")
            }
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                if object["status"] as? String == "failed" { throw NotStarted(description: object["failure"] as? String ?? "failed") }
                throw CLIError((object["error"] as? [String: Any])?["message"] as? String ?? "Server unavailable")
            }
            return object
        }
    }

    static let prompts = [
        "Write a story of about 250 words about a paper crane that learns to fly.",
        "Explain in about 200 words how a refrigerator keeps food cold.",
    ]
    /// About 18 tokens a line (75 lines: about 1,400) with a fresh first line, so no saved prompt matches it.
    static func document(lines: Int) -> String {
        "Reference \(UUID().uuidString).\n"
            + (0..<lines).map { "Line \($0): the warehouse inventory lists \($0 * 7 % 101) crates of item \($0)." }.joined(separator: " ")
            + "\nHow many crates does line \(min(41, max(0, lines - 1))) list? Answer with the number only."
    }

    static func pressureLevel() -> Int32 {
        var level: Int32 = 1
        var size = MemoryLayout<Int32>.size
        return sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &size, nil, 0) == 0 ? level : 1
    }

    /// `origami tune`: the memory measurement (`--memory` says the same), or with `--tradeoffs` the quality levels' speeds.
    static func run(flags: [String: [String]], memory: Bool) async throws {
        let settings = try Settings.load(flags: flags)
        // Strict: a value that is not a number or true/false is a usage error.
        let minutesText = flags["minutes"]?.last
        func flag(_ name: String) throws -> Bool {
            guard let text = flags[name]?.last else { return false }
            guard let value = Settings.parseBool(text) else { throw CLIError.usage("--\(name) is true or false") }
            return value
        }
        let tradeoffs = try flag("tradeoffs") && !memory
        let givenMinutes = try minutesText.map { text -> Double in
            guard let value = Double(text), value.isFinite else { throw CLIError.usage("--minutes is a number of minutes, such as 8") }
            guard (3...(tradeoffs ? 30 : 60)).contains(value) else {
                throw CLIError.usage(tradeoffs ? "--minutes is 3 to 30 with --tradeoffs" : "--minutes is 3 to 60")
            }
            return value
        }
        try Commands.requireReady(settings)
        let client = Client(base: URL(string: "http://127.0.0.1:\(settings.port)")!, key: settings.apiKey)
        // From the command line, the dashboard's rule and words (a job the server started has passed them already).
        if ProcessInfo.processInfo.environment["ORIGAMI_JOB"] == nil,
            let block = ((try? await client.call("v1/dashboard"))?["dashboard"] as? [String: Any])?["tune_block"] as? String, !block.isEmpty
        {
            throw CLIError(block)
        }
        let status = try await client.call("v1/status")
        guard let tuning = status["tuning"] as? [String: Any], let key = tuning["key"] as? String,
            let identity = tuning["identity"] as? [String: Any]
        else { throw CLIError("The model on port \(settings.port) cannot be tuned (it runs an older Origami)") }
        guard tradeoffs else {
            return try await measureMemory(
                settings, client: client, status: status, key: key, identity: identity, minutes: givenMinutes ?? 30)
        }
        try await explore(
            settings, client: client, key: key, minutes: givenMinutes ?? 8, offered: status["quality_levels"] as? [String] ?? ["exact"],
            floor: flags["quality-floor"]?.last, ownership: RequestOwnership(status))
    }

    static func record(_ settings: Settings, key: String, _ fields: [String: Any]) {
        var line = fields
        line["key"] = key
        line["date"] = ISO8601DateFormatter().string(from: Date())
        guard let data = try? JSONSerialization.data(withJSONObject: line, options: [.sortedKeys]) else { return }
        let url = historyFile(settings)
        if !FileManager.default.fileExists(atPath: url.path) { FileManager.default.createFile(atPath: url.path, contents: nil) }
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(data + Data("\n".utf8))
            try? handle.close()
        }
    }
}
