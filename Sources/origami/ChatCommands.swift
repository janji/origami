import Foundation
import OrigamiEngine

/// What `origami save`, `forget` and `prompts` (and the dashboard's `/v1/prompts`) need from the model that runs,
/// whichever engine it is: its saved prompts, its tokenizer, saving a chat's system prompt, and waiting for saves in flight.
protocol PromptKeeper {
    var savedPrompts: PromptCache? { get }
    var tokenizer: TokenizerStack { get }
    func saveSystemPrompt(_ request: ServingRequest, name: String?, cancellation: GenerationControl) throws -> String
    func drainSaves(cancellation: GenerationControl?) throws
}
/// Qwen's loaded runtime as a `PromptKeeper`.
struct RuntimeKeeper: PromptKeeper {
    let runtime: OrigamiRuntime.Loaded
    var savedPrompts: PromptCache? { runtime.promptCache }
    var tokenizer: TokenizerStack { runtime.backend.tokenizer }
    func saveSystemPrompt(_ request: ServingRequest, name: String?, cancellation: GenerationControl) throws -> String {
        try runtime.backend.saveSystemPrompt(request, name: name, cancellation: cancellation)
    }
    func drainSaves(cancellation: GenerationControl?) throws { try runtime.backend.drainSaves(cancellation: cancellation) }
}
/// The family engines save as they go: nothing is in flight.
extension FamilyServingBackend: PromptKeeper {
    var savedPrompts: PromptCache? { promptCache }
    func drainSaves(cancellation: GenerationControl?) throws {}
}

/// Answers `origami <verb>` typed as a whole chat message (see `ChatCommand`).
enum ChatCommands {
    static let help = """
        commands (type the whole message):
        - `origami status`: version, model, memory, stores
        - `origami save [name]`: keep this chat's system prompt cached, pinned under a name
        - `origami prompts`: the saved system prompts
        - `origami forget <name|all>`: remove saved system prompts
        - `origami quality [level]`: the quality level for chats with this system prompt (without a level, the current one and this model's levels)
        - `origami fast on|off`: the same as `origami quality faster` or `exact`
        - `origami help`: this list
        """

    /// What each non-exact level costs, as the chat reply says it (measured on an M2 Pro with 32 GB at 8K).
    static let effect: [String: String] = [
        "faster": "decode about 1.7× as fast, replies differ slightly",
        "fastest": "decode about 2.1× as fast, replies differ more and may slip on strict instructions",
        "dynamic-fastest": "decode about 1.4× as fast, fastest's shortcut only after tokens the model is sure of; replies differ slightly",
    ]

    /// The handler `LocalHTTPConfiguration.commands` calls: `report` is `/v1/status`'s, `serving` the backend's status,
    /// `keeper` the running model's saved prompts (nil while it loads), `family` the served model's (its levels).
    static func handler(
        report: @escaping @Sendable () -> [String: Any], serving: @escaping @Sendable () -> ServingStatus,
        keeper: @escaping @Sendable () -> PromptKeeper?, sticky: StickyQuality, defaultQuality: QualityDial, family: ModelFamily
    )
        -> @Sendable (ChatCommand, ServingRequest, GenerationControl) throws -> String
    {
        return { command, request, control in
            switch command.verb {
            case "help": return help
            case "status": return status(report(), serving())
            case "fast", "quality":
                guard let key = request.systemKey else {
                    throw ServingFailure("`origami \(command.verb)` is kept per system prompt, and this chat has none")
                }
                let levels = QualityDial.offered(family).map { "`\($0)`" }.joined(separator: ", ")
                guard let given = command.verb == "fast" ? (command.arguments[0] == "on" ? "faster" : "exact") : command.arguments.first
                else {
                    let current = (sticky.level(key) ?? defaultQuality).offered(by: family)
                    return
                        "quality **\(current.name)** for chats with this system prompt (levels: \(levels); `origami quality <level>` changes it)."
                }
                // A level this model does not offer is kept as the one it runs.
                guard let name = QualityDial.level(given)?.offered(by: family).name else { return "quality is one of \(levels)." }
                guard name == "exact" || serving().fastOptIn else { return "only exact is allowed on this server (`allow_fast` is false)." }
                try sticky.set(key, name)
                return name == "exact"
                    ? "quality **exact** for chats with this system prompt: the model's own replies."
                    : "quality **\(name)** for chats with this system prompt (\(Self.effect[name] ?? "faster decode, replies differ slightly"); `origami quality exact` undoes it)."
            default:
                guard let keeper = keeper() else {
                    throw ServingFailure("The model is still loading; try again in a moment", status: 503, type: "engine_unavailable")
                }
                guard let cache = keeper.savedPrompts else { throw ServingFailure("Saved prompts are off on this server") }
                switch command.verb {
                case "save": return try keeper.saveSystemPrompt(request, name: command.arguments.first, cancellation: control)
                case "forget":
                    try keeper.drainSaves(cancellation: control)
                    let removed = cache.forget(named: command.arguments[0].lowercased() == "all" ? nil : command.arguments[0])
                    return removed.isEmpty
                        ? "nothing to forget (`origami prompts` lists the saved ones)."
                        : "forgot " + removed.map { "**\($0)**" }.joined(separator: ", ") + "."
                default:
                    try keeper.drainSaves(cancellation: control)
                    let saved = cache.systemSnapshots
                    let kept = cache.keptSessions
                    guard !saved.isEmpty || !kept.isEmpty else {
                        return "no saved system prompts yet; they are saved automatically, and `origami save <name>` pins this chat's."
                    }
                    let ago = RelativeDateTimeFormatter()
                    ago.locale = Locale(identifier: "en_US_POSIX")
                    var lines =
                        saved.isEmpty
                        ? []
                        : ["saved system prompts:"]
                            + saved.map { item in
                                "- **\(item.name ?? "unnamed")** · \(item.tokens) tokens\(item.pinned ? " · pinned" : "") · used "
                                    + ago.localizedString(for: item.used, relativeTo: Date()) + " · " + OrigamiPlan.gigabytes(item.bytes)
                            }
                    if !kept.isEmpty {
                        lines +=
                            ["kept sessions (X-Origami-Keep):"]
                            + kept.map {
                                "- **\($0.name)** · \($0.tokens) tokens · used " + ago.localizedString(for: $0.used, relativeTo: Date())
                            }
                    }
                    return lines.joined(separator: "\n")
                }
            }
        }
    }

    /// `/v1/prompts` (`LocalHTTPConfiguration.prompts`) for either engine: GET one saved prompt with its tokens decoded
    /// (and for a system prompt its own text, to edit), PATCH its `name` or `keep`, DELETE it, and POST a system prompt's
    /// `text` saved under `name` as `origami save` saves a chat's; with `replace`, the entry it was edited from goes once
    /// the new one is saved.
    static func prompts(keeper: @escaping @Sendable () -> PromptKeeper?)
        -> @Sendable (String, String?, [String: Any], ServingRequest?, GenerationControl) throws -> [String: Any]
    {
        return { method, id, body, save, control in
            guard let keeper = keeper() else {
                throw ServingFailure("The model is still loading; try again in a moment", status: 503, type: "engine_unavailable")
            }
            guard let cache = keeper.savedPrompts else { throw ServingFailure("Saved prompts are off on this server", status: 404) }
            switch (method, id, save) {
            case ("GET", let id?, _):
                let (item, tokens) = try cache.saved(id)
                let tokenizer = keeper.tokenizer
                var row = PromptCache.wire(item)
                let text = try tokenizer.decode(tokens)
                row["text"] = text
                if item.kind == "system",
                    let probe = try? tokenizer.decode(
                        tokenizer.encodeChat(messages: [
                            ["role": .string("system"), "content": .string("\u{2}")],
                            ["role": .string("user"), "content": .string("\u{1}")],
                        ]))
                {
                    let own = PromptCache.systemText(text, probe: probe)
                    row["system_text"] = own.text
                    row["system_whole"] = own.whole
                }
                return row
            case ("PATCH", let id?, _):
                return PromptCache.wire(try cache.update(id, name: body["name"] as? String, keep: body["keep"] as? Bool))
            case ("DELETE", let id?, _):
                try keeper.drainSaves(cancellation: control)
                try cache.remove(id)
                return ["deleted": id]
            case ("POST", nil, let save?):
                guard let name = body["name"] as? String else {
                    throw ServingFailure("POST {\"text\": \"<system prompt>\", \"name\": \"<name>\"}")
                }
                let replace = body["replace"] as? String
                guard !cache.entries.contains(where: { $0.name == name && $0.id != replace }) else {
                    throw ServingFailure("Another saved prompt is named \(name)", status: 409)
                }
                try keeper.drainSaves(cancellation: control)
                let before = Set(cache.entries.map(\.id))
                let message = try keeper.saveSystemPrompt(save, name: name, cancellation: control)
                // The entry saved: the new one under `name`, else the one that already held this text.
                let named = cache.entries.filter { $0.kind == "system" && $0.name == name }.map(\.id)
                guard let saved = named.first(where: { !before.contains($0) }) ?? named.first(where: { $0 != replace }) ?? named.first
                else {
                    throw ServingFailure("The system prompt was read but not kept; try again", status: 409)
                }
                if let replace, replace != saved { try? cache.remove(replace) }
                return ["id": saved, "message": message.replacingOccurrences(of: "**", with: "")]
            default: throw ServingFailure("Use GET, PATCH or DELETE /v1/prompts/<id>, or POST /v1/prompts", status: 405)
            }
        }
    }

    static func status(_ report: [String: Any], _ status: ServingStatus) -> String {
        var lines = [
            "Origami \(status.version), \(status.ready ? "ready" : (status.failure == nil ? "loading" : "failed")), \(status.contextTier / 1024)K context"
                + (status.fastOptIn ? ", quality levels allowed" : ", exact only")
        ]
        if let model = report["model"] as? [String: Any] { lines.append("model \(model["name"] ?? "?") (\(model["quantization"] ?? "?"))") }
        if let memory = report["memory"] as? [String: Any], let budget = memory["budget_bytes"] as? UInt64,
            let cap = memory["cap_bytes"] as? UInt64
        {
            lines.append("memory budget \(OrigamiPlan.gigabytes(budget)) of up to \(OrigamiPlan.gigabytes(cap))")
        }
        if let cache = report["expert_cache"] as? [String: Any], let rate = cache["hit_rate"] as? Double {
            lines.append(String(format: "expert cache %@ experts, %.0f%% hits", "\(cache["slots"] ?? 0)", rate * 100))
        }
        if let stores = report["stores"] as? [[String: Any]] {
            let used = stores.filter { store in ["pack_in_use", "copy_in_use", "hot_copy_in_use"].contains { store[$0] as? Bool == true } }
                .map { "\($0["drive"] ?? "?")" }
            lines.append(
                "reads from " + (used.isEmpty ? "the model files" : used.joined(separator: " and ")) + " (\(report["reads"] ?? "?"))")
        }
        if let entries = (report["prompt_cache"] as? [String: Any])?["entries"] as? [[String: Any]] {
            lines.append("saved prompts: \(entries.filter { $0["kind"] as? String == "system" }.count) system, \(entries.count) in all")
        }
        if let note = status.degraded { lines.append("note: \(note)") }
        return "status\n" + lines.map { "- " + $0 }.joined(separator: "\n")
    }
}

/// `origami quality <level>` (and `origami fast on|off`) per system prompt, kept in the data folder as level names
/// (the 2.0 file held `true` for fast, still read).
final class StickyQuality: @unchecked Sendable {
    private let file: URL
    private let lock = NSLock()
    private var levels: [String: String]
    init(file: URL) {
        self.file = file
        let data = try? Data(contentsOf: file)
        levels =
            data.flatMap { try? JSONDecoder().decode([String: String].self, from: $0) }
            ?? data.flatMap { try? JSONDecoder().decode([String: Bool].self, from: $0) }?.compactMapValues { $0 ? "faster" : nil } ?? [:]
    }
    func level(_ key: String) -> QualityDial? {
        lock.lock()
        defer { lock.unlock() }
        return levels[key].flatMap(QualityDial.level)
    }
    /// Writes under the lock, so two commands at once cannot leave the older state on disk.
    func set(_ key: String, _ name: String) throws {
        lock.lock()
        defer { lock.unlock() }
        levels[key] = name
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(levels).write(to: file, options: .atomic)
    }
}
