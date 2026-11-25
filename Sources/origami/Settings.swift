import Foundation
import OrigamiEngine
import Security

/// User settings. Each one comes from, in order: a command-line flag, an `ORIGAMI_*`
/// environment variable, the config file, or an automatic default. A plain run needs none.
struct Settings {
    /// One store folder from `stores`, with the most bytes Origami may write there.
    struct Store: Codable, Equatable {
        var path: String
        /// Most bytes Origami may write there; nil means "what fits".
        var size: UInt64?
    }
    /// A local folder or a Hugging Face repo id (`org/name`).
    var model: String?
    var stores: [Store] = []
    var dataDir: URL
    var memoryLimit: UInt64?
    /// Without memory_limit: an idle server measures the fastest memory limit itself when none is current (`AutoMeasure`).
    var memoryAutoMeasure = false
    /// After `origami stop` (or an idle unload), a request starts the model instead of getting 503 `stopped`.
    var startOnRequest = false
    /// Unload the model after this many minutes without a request, as a stop does (nil: never).
    var idleUnloadMinutes: Double?
    /// The cached experts' share of memory; nil leaves them what memory_limit allows.
    var expertMemory: UInt64?
    /// Folder for a context kept on disk and the saved prompts; nil uses the data folder (`promptCacheRoot`).
    var contextDir: URL?
    /// False when the data folder's drive has no file ownership (exFAT, FAT): prompts can be neither kept private nor
    /// reused there, so without `context_dir` they go to the default data folder on the Mac's own drive.
    var promptsBesideData = true
    /// The copy of the most-used experts next to the model: nil automatic, 0 off, else its size in bytes.
    var hotCopy: UInt64?
    /// hot_copy presets, the same on every engine: save-space is off, balanced the automatic size (nil), and fastest
    /// asks for every expert; each engine's placement caps it at what the drive holds above its free-space reserve.
    static let hotCopyFastest = UInt64.max
    static func hotCopyPreset(_ text: String) -> UInt64? {
        switch text {
        case "off", "save-space": 0
        case "fastest": hotCopyFastest
        default: nil
        }
    }
    static func hotCopyName(_ bytes: UInt64) -> String {
        bytes == 0 ? "off" : bytes == hotCopyFastest ? "fastest" : OrigamiPlan.gigabytes(bytes)
    }
    /// Folder for that copy (either engine's hot copy); nil: next to the model (Qwen), or the data folder.
    var hotCopyDir: URL?
    /// A further place for the hot copy (`hot_copy_more`): a folder on another drive and its size there (nil automatic,
    /// 0 off, else bytes; `hotCopyFastest` as much as the drive holds above its reserve).
    struct CopyPlace: Equatable {
        var folder: URL
        var size: UInt64?
    }
    /// The places after `hot_copy_dir`'s, each on its own drive: the model's experts are split over all of them, the
    /// fastest drive holding the most-used.
    var hotCopyMore: [CopyPlace] = []
    /// `hot_copy_keep`: with a `hot_copy` size set, this model's copy keeps that size while another model is loaded
    /// (`IdleHotCopy`); off (the default), it gives the loaded model's copy what lies above its floor.
    var hotCopyKeep = false
    /// More folders the model search (`Models.found`) looks in.
    var modelsDirs: [URL] = []
    var context = 8192
    var port = 11439
    var allowFast = true
    /// The quality dial of requests that name no level: `quality`'s level with any knob set alone on top.
    var quality = QualityDial.exact
    /// Why a stored non-exact `quality` serves exact (allow_fast false), for the server's log.
    var qualityNote: String?
    /// Effort for requests that send no thinking control; nil keeps the server's default (none).
    var reasoningEffort: String?
    /// Thinking budget for requests that send none; nil leaves only the answer room.
    var maxThinkingTokens: Int?
    /// `origami <verb>` chat messages are commands.
    var commands = true
    /// List every model with its quality levels and switch by name (`list_models`); otherwise one model, `origami`.
    var listModels = false
    /// Stopped and unloaded (`origami stop`, the dashboard's Stop and unload): the front serves the dashboard and loads
    /// no model until `origami start`.
    var stopped = false
    /// GLM and DeepSeek: their context's per-position state in files on the context drive (true), in memory (false), or
    /// as its size decides (nil, `auto`).
    var contextOnDisk: Bool?
    /// Streamed replies carry status lines as reasoning while they wait or read a long prompt.
    var statusMessages = false
    /// Qwen: the GPU keep-alive spin while a reply is made, on (true), off (false), or as the plan decides (nil, `auto`).
    var gpuKeepAlive: Bool?
    /// Opt-in MTP speculation: the safetensors file with the model's `language_model.mtp.*` head, or nil for off.
    var mtpWeights: URL?
    /// Request records: a folder for one JSON line per request, or nil for none.
    var requestLog: URL?
    /// The key every request to the server must carry, or nil for none (the default).
    var apiKey: String?
    /// `listen lan`: the front listens on every interface (its model servers stay on 127.0.0.1); `local` (false, the
    /// default): on 127.0.0.1 only.
    var listenLAN = false
    /// The key other devices on the network send when no `api_key` is set (`lan_key`; generated when `listen lan` is
    /// set), or nil; `lanOpen` (`lan_key off`): they need none.
    var lanKey: String?
    var lanOpen = false
    /// The key a request from another device must carry, or nil for none: `api_key` (every client needs it), else
    /// `lan_key` unless it is off. With `lan_key` missing (never generated) the front refuses other devices.
    var remoteKey: String? { apiKey ?? (lanOpen ? nil : lanKey) }
    /// `trusted_proxy loopback`: a connection from 127.0.0.0/8 that carries `X-Forwarded-For` comes from the client
    /// that header names (a TLS proxy on this Mac), which then counts as another device; off (false, the default): the
    /// header is ignored.
    var trustedProxy = false
    /// Saved prompts kept above an 8K context (8K keeps eight).
    var promptCacheEntries = PromptCache.defaultEntries
    /// `expert_bits` as a flag, variable, the settings file or a named model still sets it ("3 (settings file)"), or nil.
    /// The option is gone (published low-bit experts replace it); the model's own experts run and `retiredExpertBits`
    /// says so at start, in `status` and in `doctor`.
    var retiredExpertBits: String?
    /// `sampling custom`: `samplingValues` (keys of `SamplingDefaults.keys`) over the model's recommendation; else the
    /// recommendation alone (the values stay saved for a later custom).
    var samplingCustom = false
    var samplingValues: [String: Double] = [:]
    /// `sampling_override`: these defaults even over a request's own values.
    var samplingOverride = false
    /// A request's defaults for a model that recommends `recommended`.
    func sampling(_ recommended: SamplingDefaults) -> SamplingDefaults {
        samplingCustom && !samplingValues.isEmpty ? recommended.setting(samplingValues, source: "custom") : recommended
    }
    /// Settings kept per model (in its entry in models.json, which beats the file's key) when the model is a named one.
    static let modelKeys =
        [
            "context", "quality", "reasoning_effort", "memory_limit", "hot_copy", "hot_copy_dir", "hot_copy_more", "hot_copy_keep",
            "sampling",
        ]
        + SamplingDefaults.keys + ["sampling_override"] + qualityKnobs
    /// When a change applies to a running server (`origami config set` says it; the dashboard's "reload required"
    /// compares every `needsReload` key the loaded model runs with): read afresh (`appliesNow`), read once as the server
    /// starts (`appliesAtRestart`), every other key when the model next loads. `model` is told apart by the front itself.
    static let appliesNow: Set<String> = [
        "memory_auto_measure", "lan_key", "models_dirs", "hot_copy_keep", "stopped", "start_on_request", "idle_unload_minutes",
    ]
    static let appliesAtRestart: Set<String> = ["port", "listen", "data_dir", "trusted_proxy"]
    static var needsReload: [String] {
        keys.map(\.0).filter { !appliesNow.contains($0) && !appliesAtRestart.contains($0) && $0 != "model" }
    }
    /// Why an `expert_bits` setting no longer does anything, and what replaces it.
    static let expertBitsRetired =
        "is no longer an option: Origami runs the model's own experts. For smaller, faster experts add a published "
        + "low-bit experts file: `origami model add <experts.gguf> --base <model>` (see docs/models.md)"
    /// The line start, `status`, `doctor` and `setup` show while `expert_bits` is still set, or nil.
    var expertBitsNote: String? {
        retiredExpertBits.map {
            "expert_bits \($0) \(Self.expertBitsRetired). `origami config unset expert_bits` removes the setting."
        }
    }
    /// Where each key's value came from: "flag", "env", "file" or "default" (absent).
    var sources: [String: String] = [:]
    /// Keys in the settings file that no setting reads (a typo, or one a later version dropped).
    var unknownKeys: [String] = []
    /// The warning `config` and `doctor` print for `unknownKeys`, or nil.
    var unknownKeysNote: String? {
        unknownKeys.isEmpty
            ? nil
            : "\(configFile.path) holds \(unknownKeys.joined(separator: ", ")), which no setting reads: check the spelling "
                + "(`origami config` lists the settings), or `origami config unset <key>` to remove it"
    }
    /// Where the config file lives (`ORIGAMI_CONFIG` or `--config` select another, e.g. one per port).
    let configFile: URL

    static let defaultDataDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/Origami", isDirectory: true)
    static let logDir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/Origami", isDirectory: true)
    /// Keys `origami config` understands, with their meaning for `origami config` without arguments.
    static let keys: [(String, String)] = [
        ("model", "model folder, Hugging Face repo id, or a name from `origami model list`"),
        (
            "stores",
            "folders where packed experts may be written, comma-separated (see `origami store`); the data folder is always used"
        ),
        (
            "data_dir",
            "Origami's settings, usage counts, hot copy, saved prompts and downloads (default ~/Library/Application Support/Origami)"
        ),
        ("memory_limit", "memory ceiling, e.g. 20GB; lowered while the Mac cannot keep it (default: automatic)"),
        (
            "memory_auto_measure",
            "without memory_limit: the server finds the fastest memory limit by itself (as `origami tune --memory` does) when none is measured for this model and context or it went stale, after half an hour without requests, on power, at most once a day; a request stops it: true or false (default false)"
        ),
        (
            "start_on_request",
            "after `origami stop` or an idle unload, a request starts the model (it waits for the load) instead of getting \"Origami is stopped\": true or false (default false)"
        ),
        (
            "idle_unload_minutes",
            "unload the model after this many minutes without a request, giving its memory back as `origami stop` does; with start_on_request the next request loads it again: minutes, or off (the default)"
        ),
        (
            "context",
            "context length in tokens, up to 262144: 8k, 32k, 100000, 256k… (default 8k). Above what this Mac keeps in memory (`origami setup` shows it) the key/value cache is kept in a file on disk: slower, and it needs 32 GB of RAM and about 37 GB of free disk"
        ),
        ("port", "HTTP port (default 11439), on 127.0.0.1, or on every interface with listen lan"),
        (
            "listen",
            "who can reach Origami: local (the default, this Mac only, on 127.0.0.1) or lan (also other devices on this network, at this Mac's name or address; they need lan_key). Takes effect when origami serve starts"
        ),
        (
            "lan_key",
            "the key other devices on the network send with listen lan (Authorization: Bearer <key>, or #key=<key> after the dashboard's address), made when listen lan is set; programs on this Mac need none unless api_key is set, which every client then needs. off: other devices need no key, so anything on the network can use Origami and change its settings (less safe). Never shown here: the dashboard on this Mac shows it, with a QR code"
        ),
        (
            "trusted_proxy",
            "loopback: a TLS proxy on this Mac (Caddy, nginx) forwards other devices' requests, and X-Forwarded-For from 127.0.0.1 names each client, which then needs api_key (see docs/integration.md); off (the default): that header is ignored"
        ),
        (
            "allow_fast",
            "allow quality levels other than exact (per request, the -<level> model ids or `quality`): true or false (default true)"
        ),
        (
            "quality",
            "quality dial for requests that choose none: " + QualityDial.offeredByModel
                + " (default exact, the model's own replies; the others decode faster and reply slightly differently; a level the model does not offer runs as its nearest one). Setting it resets the five knobs below"
        ),
        (
            "experts_per_token",
            "knob: limit routed experts per token, 1 to 10, or all (default: the level's); selection follows the model engine's routing policy"
        ),
        ("tail_skip", "knob: drop routed experts under this share of the gate weight, e.g. 0.05, or off (default: the level's)"),
        ("tail_skip_misses_only", "knob: the tail skip drops only experts not in memory: true or false (default: the level's)"),
        (
            "late_miss",
            "knob: an expert still being read when needed: wait, skip or substitute (the best one in memory) (default: the level's)"
        ),
        (
            "confident_above",
            "knob: the other knobs apply to a decode step only after a token the model gave at least this probability, exact after the others, e.g. 0.9, or off (default: the level's)"
        ),
        ("reasoning_effort", "thinking for requests that ask for none: \(ServingDefaults.efforts.joined(separator: ", ")) (default none)"),
        ("max_thinking_tokens", "close the think block after this many tokens when a request sets no budget (default: none)"),
        (
            "sampling",
            "how replies are sampled when an app sends no values of its own: recommended (the model's own recommendation, from its generation_config.json or GGUF, else Qwen's; the default) or custom (the five values below over it). Kept per model"
        ),
        (
            "temperature",
            "custom sampling: creativity, 0 (always the most likely word, the same reply every time) to 2 and more (more varied)"
        ),
        ("top_p", "custom sampling: word pool, the share of probability the next word is drawn from, above 0 to 1 (1 = every word)"),
        ("top_k", "custom sampling: shortlist, the most likely words the next one is drawn from, 0 (no limit) or more"),
        ("min_p", "custom sampling: minimum chance, a word's least probability against the likeliest one's, 0 to 1"),
        ("presence_penalty", "custom sampling: repetition brake, -2 to 2; above 0 makes words already used less likely"),
        (
            "sampling_override",
            "use these sampling values even when an app sends its own: true or false (default false, the app's values win). Kept per model"
        ),
        ("commands", "answer `origami <verb>` chat messages: true or false (default true)"),
        (
            "context_on_disk",
            "GLM-5.3 and DeepSeek-V4.1: keep the context in files on the context drive (true), in memory (false), or auto (the default: on disk when it would take more than a quarter of what Origami may use)"
        ),
        (
            "list_models",
            "list one model, `origami`, the setup chosen in the dashboard (false, the default), or every model with its quality levels, a request switching between them by name (true)"
        ),
        (
            "stopped",
            "true while Origami is stopped and its model unloaded (origami stop, or Stop and unload in the dashboard): the server keeps the dashboard up, loads no model and answers apps that it is stopped, until origami start or the dashboard's Start (default false)"
        ),
        ("status_messages", "show progress of long prompts as thinking lines in streamed replies: true or false (default false)"),
        (
            "gpu_keepalive",
            "keep the GPU busy with a tiny spin command while a reply is made, so its clock does not drop between Origami's short commands or while experts are read: true, false or auto (the default: on, except GLM and DeepSeek in Low Power Mode). Replies are the same either way; if replies are unusually slow on this Mac, try false (another engine's keep-alive made decode about 19 times slower on a MacBook Air M5 and a Mac Studio M1 Max)"
        ),
        (
            "speculation",
            "MTP speculative decoding: off, or a safetensors file with the model's multi-token-prediction head (`language_model.mtp.*`, MLX 4-bit; ARC4NUM's MLX-Serve build ships it). A potential speed upgrade for Macs that hold every expert in memory (96 GB or more); Origami measures each reply and decodes plainly where it does not pay (default off)"
        ),
        (
            "request_log",
            "folder for a JSON line per request (sizes, timings, cache share; system prompts and tools stored once by hash); off unless set"
        ),
        (
            "api_key",
            "a key every request must carry (Authorization: Bearer <key> or x-api-key: <key>), 16 to 256 characters without spaces, such as `openssl rand -hex 16` prints; `origami dashboard` brings it along. Off (default): any program on this Mac that reaches the port can use Origami. Never shown, only whether it is set"
        ),
        (
            "expert_memory",
            "memory for cached experts, e.g. 12GB (default: what memory_limit leaves after Origami's fixed parts); less leaves more to other apps and makes replies slower"
        ),
        (
            "context_dir",
            "folder for a context kept on disk and for saved prompts (default <data_dir>/prompt-cache, or this Mac's own data folder when data_dir is on a drive without file ownership, such as exFAT); the fastest drive with room is best"
        ),
        (
            "prompt_cache_entries",
            "saved prompts and conversations kept on disk above an 8K context, \(PromptCache.entryRange.lowerBound) to \(PromptCache.entryRange.upperBound) "
                + "(default \(PromptCache.defaultEntries); 8K keeps 8). " + promptCacheDisk
                + " More entries let more chats restart from where they were; "
                + "saves always leave 10 GB of the drive free, dropping the oldest entries"
        ),
        (
            "hot_copy",
            "disk space for a copy of the most-used experts, which reads faster than the model's files: a size such as 20GB, or off; presets: save-space (off), balanced (automatic), fastest (as large as the drive allows) (default: automatic). automatic grows it while the usage counts show it pays, always leaving 20 GB of the drive free. Qwen: `origami setup` builds it; GLM and DeepSeek: built while serving, after the first reply, on a drive other than the model's"
        ),
        (
            "hot_copy_dir",
            "folder for that copy, on the fastest drive other than the pack's (default: a store on the model's drive for Qwen, the data folder for GLM and DeepSeek)"
        ),
        (
            "hot_copy_more",
            "more places for the hot copy, one per drive: folder=size pairs, comma-separated, such as /Volumes/Fast/Origami/hot-copy=200GB (size: as hot_copy, default auto), or off for none. The most-used experts go to the fastest drive, the next ones to the next drive, none stored twice; each place keeps 20 GB of its drive free (default: none)"
        ),
        (
            "hot_copy_keep",
            "with a hot_copy size set: keep that size while another model is loaded: true or false (default false: an idle model's copy gives the loaded model's copy on the same drive its least-used experts, down to those covering half of its uses)"
        ),
        (
            "models_dirs",
            "more folders the dashboard's model search looks in, comma-separated (default: none; the usual download places are always searched)"
        ),
    ]

    /// What the saved prompts take on disk at most, per entry and for each choice, at the contexts above 8K.
    static var promptCacheDisk: String {
        let contexts = [32_768, 131_072, 262_144]
        let sizes = contexts.map { PromptCache.entryDiskBytes(context: $0) }
        func list(_ count: Int) -> String {
            zip(contexts, sizes).map { "\(OrigamiPlan.gigabytes($1 * UInt64(count))) at \($0 / 1024)K" }.joined(separator: ", ")
        }
        let (fewest, most) = (PromptCache.entryRange.lowerBound, PromptCache.entryRange.upperBound)
        return "Each takes up to " + list(1) + " (a full context; a shorter prompt takes less), so \(fewest) take up to " + list(fewest)
            + " and \(most) up to " + list(most) + "."
    }

    /// A key's effective value as `origami config` shows it.
    func display(_ key: String) -> String {
        switch key {
        case "model": return model ?? "(automatic)"
        case "stores":
            return
                (stores.isEmpty
                ? "(none)" : stores.map { $0.path + ($0.size.map { " up to \(OrigamiPlan.gigabytes($0))" } ?? "") }.joined(separator: ", "))
                + "; \(homeStore.path) (always used)"
        case "data_dir": return dataDir.path
        case "memory_limit": return memoryLimit.map(OrigamiPlan.gigabytes) ?? "(fitted to this Mac)"
        case "memory_auto_measure": return "\(memoryAutoMeasure)"
        case "start_on_request": return "\(startOnRequest)"
        case "idle_unload_minutes": return idleUnloadMinutes.map { Int(exactly: $0).map { "\($0)" } ?? "\($0)" } ?? "off"
        case "context": return context % 1024 == 0 ? "\(context / 1024)k" : "\(context)"
        case "port": return "\(port)"
        case "reasoning_effort": return reasoningEffort ?? "none"
        case "max_thinking_tokens": return maxThinkingTokens.map(String.init) ?? "(none)"
        case "commands": return "\(commands)"
        case "list_models": return "\(listModels)"
        case "stopped": return "\(stopped)"
        case "context_on_disk": return contextOnDisk.map { "\($0)" } ?? "auto"
        case "status_messages": return "\(statusMessages)"
        case "gpu_keepalive": return gpuKeepAlive.map { "\($0)" } ?? "auto"
        case "speculation": return mtpWeights?.path ?? "off"
        case "request_log": return requestLog?.path ?? "(off)"
        case "api_key": return apiKey == nil ? "(off)" : "(set)"
        case "listen": return listenLAN ? "lan" : "local"
        case "lan_key": return lanOpen ? "off" : lanKey == nil ? "(none)" : "(set)"
        case "trusted_proxy": return trustedProxy ? "loopback" : "off"
        case "prompt_cache_entries": return "\(promptCacheEntries)"
        case "sampling": return samplingCustom ? "custom" : "recommended"
        case "temperature", "top_p", "top_k", "min_p", "presence_penalty":
            return samplingValues[key].map { key == "top_k" ? "\(Int($0))" : "\($0)" } ?? "(the model's)"
        case "sampling_override": return "\(samplingOverride)"
        case "hot_copy_keep": return "\(hotCopyKeep)"
        case "expert_memory": return expertMemory.map(OrigamiPlan.gigabytes) ?? "(what memory_limit leaves)"
        case "context_dir": return promptCacheRoot.path
        case "hot_copy": return hotCopy.map(Self.hotCopyName) ?? "(automatic)"
        case "hot_copy_dir": return hotCopyDir?.path ?? "(automatic)"
        case "hot_copy_more":
            return hotCopyMore.isEmpty
                ? "(none)" : hotCopyMore.map { "\($0.folder.path)=\($0.size.map(Self.hotCopyName) ?? "auto")" }.joined(separator: ",")
        case "models_dirs": return modelsDirs.isEmpty ? "(none)" : modelsDirs.map(\.path).joined(separator: ",")
        case "quality": return quality.name == "custom" ? "custom (" + quality.signature + ")" : quality.name
        case "experts_per_token", "tail_skip", "tail_skip_misses_only", "late_miss", "confident_above": return quality.knobs[key] ?? ""
        default: return "\(allowFast)"
        }
    }
    /// Every key with its value, where it came from and what it means (`config --json`, /v1/status).
    var report: [String: [String: String]] {
        Dictionary(
            uniqueKeysWithValues: Self.keys.map { ($0.0, ["value": display($0.0), "source": sources[$0.0] ?? "default", "meaning": $0.1]) })
    }

    var promptCacheRoot: URL {
        contextDir ?? (promptsBesideData ? dataDir : Self.defaultDataDir).appendingPathComponent("prompt-cache", isDirectory: true)
    }
    var modelsDir: URL { dataDir.appendingPathComponent("models", isDirectory: true) }
    /// The model's routing counts, which rank its hot copy and further copies.
    func routingFile(_ model: ModelSource) -> URL { dataDir.appendingPathComponent("routing-\(model.fingerprint.prefix(12)).json") }
    /// The store every install has: the data folder, normally on the internal drive.
    var homeStore: URL { dataDir.appendingPathComponent("store", isDirectory: true) }
    var calibrationFile: URL { dataDir.appendingPathComponent("calibration.json") }
    var driveSpeedsFile: URL { dataDir.appendingPathComponent("drive-speeds.json") }
    /// Where expert copies may go, in the order they are tried: the stores, the hot copy's folders, the data folder's store.
    var storeFolders: [URL] {
        var folders = stores.map { URL(fileURLWithPath: $0.path, isDirectory: true) }
        for folder in [hotCopyDir].compactMap({ $0 }) + hotCopyMore.map(\.folder) + [homeStore] where !folders.contains(folder) {
            folders.append(folder)
        }
        return folders
    }
    /// Whether a pack may be built in `store`: any of `storeFolders` but a hot copy's folder (`hot_copy_dir`,
    /// `hot_copy_more`) that is not also a store, which holds the hot copy alone (a pack there would also share its drive
    /// with the copy).
    func holdsPacks(_ store: URL) -> Bool {
        !([hotCopyDir].compactMap { $0 } + hotCopyMore.map(\.folder)).contains(store) || store == homeStore
            || stores.contains { $0.path == store.path }
    }

    /// The settings file a command reads: `--config`, else `ORIGAMI_CONFIG`, else the default data folder's.
    static func configFile(flags: [String: [String]], environment: [String: String] = ProcessInfo.processInfo.environment) -> URL {
        (flags["config"]?.last ?? environment["ORIGAMI_CONFIG"]).map { URL(fileURLWithPath: expand($0)) }
            ?? defaultDataDir.appendingPathComponent("config.json")
    }

    /// Resolves settings for a command. `flags` are already-parsed `--key value` options. A value it refuses from the
    /// settings file or a model's entry names where it is.
    static func load(flags: [String: [String]], environment: [String: String] = ProcessInfo.processInfo.environment) throws -> Settings {
        var last: (key: String, place: String?)?
        do { return try load(flags: flags, environment: environment, last: &last) } catch let error as CLIError {
            guard let last, let place = last.place, error.description.hasPrefix(last.key) else { throw error }
            var named = CLIError("\(error.description) (\(last.key) in \(place))")
            named.exitCode = error.exitCode
            throw named
        }
    }

    private static func load(
        flags: [String: [String]], environment: [String: String], last: inout (key: String, place: String?)?
    ) throws -> Settings {
        let configFile = configFile(flags: flags, environment: environment)
        let file = try readConfig(configFile)
        var sources: [String: String] = [:]
        var profile: [String: Any] = [:]
        var asked: Set<String> = ["stores"]
        var registryFile: String?
        func value(_ key: String) -> String? {
            asked.insert(key)
            last = (key, nil)
            if let flag = flags[key.replacingOccurrences(of: "_", with: "-")]?.last {
                sources[key] = "flag"
                return flag
            }
            if let env = environment["ORIGAMI_" + key.uppercased()] {
                sources[key] = "env"
                return env
            }
            // A null (left by `config unset` on a named model) is the default: it records no source, so it outranks nothing.
            if Self.modelKeys.contains(key), let stored = profile[key] {
                if stored is NSNull { return nil }
                sources[key] = "model"
                last = (key, registryFile)
                return (stored as? String)
                    ?? ((stored as? NSNumber).map { CFGetTypeID($0) == CFBooleanGetTypeID() ? "\($0.boolValue)" : "\($0)" }) ?? "\(stored)"
            }
            guard let stored = file[key] else { return nil }
            sources[key] = "file"
            last = (key, configFile.path)
            return (stored as? String) ?? "\(stored)"
        }
        var settings = Settings(
            dataDir: value("data_dir").map { URL(fileURLWithPath: expand($0), isDirectory: true) } ?? defaultDataDir,
            configFile: configFile)
        settings.model = value("model")
        profile = ModelRegistry.entry(settings.model, settings: settings) ?? [:]
        registryFile = "\(settings.model ?? "the model")'s entry in \(ModelRegistry.file(settings).path)"
        if let flag = flags["store"] {
            settings.stores = flag.map { Store(path: expand($0)) }
            sources["stores"] = "flag"
        } else if let env = environment["ORIGAMI_STORES"] {
            settings.stores = env.split(separator: ",").map { Store(path: expand(String($0))) }
            sources["stores"] = "env"
        } else if let list = file["stores"] as? [[String: Any]] {
            settings.stores = list.compactMap { entry in
                (entry["path"] as? String).map { Store(path: expand($0), size: (entry["size"] as? NSNumber)?.uint64Value) }
            }
            sources["stores"] = "file"
        }
        if let text = value("memory_limit"), text != "auto" {
            guard let bytes = parseBytes(text) else { throw CLIError("memory_limit \"\(text)\" is not a size such as 20GB") }
            settings.memoryLimit = bytes
        }
        if let text = value("expert_memory"), text != "auto" {
            guard let bytes = parseBytes(text) else { throw CLIError("expert_memory \"\(text)\" is not a size such as 12GB") }
            settings.expertMemory = bytes
        }
        if let text = value("context_dir") { settings.contextDir = URL(fileURLWithPath: expand(text), isDirectory: true) }
        if settings.contextDir == nil {
            let folder = settings.dataDir
            settings.promptsBesideData = Folders.timed(2, on: folder.path) { PromptCache.driveKeepsOwnership(folder) } ?? true
        }
        if let text = value("hot_copy_dir"), text != "auto" {
            guard !isOff(text) else { throw CLIError(hotCopyDirOff) }
            settings.hotCopyDir = URL(fileURLWithPath: expand(text), isDirectory: true)
        }
        if let text = value("hot_copy_more") { settings.hotCopyMore = try copyPlaces(text) }
        if let text = value("models_dirs") { settings.modelsDirs = list(text).map { URL(fileURLWithPath: $0, isDirectory: true) } }
        if let text = value("hot_copy"), text != "auto", text != "balanced" {
            guard let bytes = Self.hotCopyPreset(text) ?? parseBytes(text) else {
                throw CLIError("hot_copy \"\(text)\" is a size such as 20GB, off, auto, or save-space, balanced, fastest")
            }
            settings.hotCopy = bytes
        }
        if let text = value("context") {
            guard let tokens = parseContext(text), tokens <= OrigamiPlan.maximumContext else {
                throw CLIError("context \"\(text)\" is not a token count up to 262144 (256k)")
            }
            settings.context = tokens
        }
        if let text = value("port") {
            guard let port = Int(text), (1...65535).contains(port) else { throw CLIError("port \"\(text)\" is not a port number") }
            settings.port = port
        }
        if let text = value("allow_fast") {
            guard let flag = parseBool(text) else { throw CLIError("allow_fast \"\(text)\" is not true or false") }
            settings.allowFast = flag
        }
        if let text = value("reasoning_effort") {
            guard ServingDefaults.efforts.contains(text) else {
                throw CLIError("reasoning_effort \"\(text)\" is not one of \(ServingDefaults.efforts.joined(separator: ", "))")
            }
            settings.reasoningEffort = text
        }
        if let text = value("max_thinking_tokens") {
            guard let tokens = Int(text), tokens >= 1 else { throw CLIError("max_thinking_tokens \"\(text)\" is not a positive number") }
            settings.maxThinkingTokens = tokens
        }
        if let text = value("speculation") { settings.mtpWeights = isOff(text) ? nil : URL(fileURLWithPath: expand(text)) }
        if let text = value("request_log") {
            settings.requestLog = isOff(text) ? nil : URL(fileURLWithPath: expand(text), isDirectory: true)
        }
        if let text = value("prompt_cache_entries") {
            guard let count = Int(text), PromptCache.entryRange.contains(count) else {
                throw CLIError(
                    "prompt_cache_entries \"\(text)\" is not \(PromptCache.entryRange.lowerBound) to \(PromptCache.entryRange.upperBound)")
            }
            settings.promptCacheEntries = count
        }
        if let text = value("api_key"), !isOff(text) {
            guard isAPIKey(text) else {
                throw CLIError(
                    "api_key must be 16 to 256 characters without spaces (from \(sources["api_key"] ?? "file")); `origami config set api_key off` turns it off"
                )
            }
            settings.apiKey = text
        }
        if let text = value("listen") {
            guard ["local", "lan"].contains(text.lowercased()) else { throw CLIError("listen \"\(text)\" is local or lan") }
            settings.listenLAN = text.lowercased() == "lan"
        }
        if let text = value("lan_key") {
            if isOff(text) {
                settings.lanOpen = true
            } else {
                guard isAPIKey(text) else {
                    throw CLIError(
                        "lan_key must be 16 to 256 characters without spaces (from \(sources["lan_key"] ?? "file")); `origami config set lan_key on` makes a new one"
                    )
                }
                settings.lanKey = text
            }
        }
        if let text = value("trusted_proxy") {
            guard isOff(text) || text.lowercased() == "loopback" else { throw CLIError("trusted_proxy \"\(text)\" is off or loopback") }
            settings.trustedProxy = !isOff(text)
        }
        if let text = value("memory_auto_measure") {
            guard let flag = parseBool(text) else { throw CLIError("memory_auto_measure \"\(text)\" is not true or false") }
            settings.memoryAutoMeasure = flag
        }
        if let text = value("start_on_request") {
            guard let flag = parseBool(text) else { throw CLIError("start_on_request \"\(text)\" is not true or false") }
            settings.startOnRequest = flag
        }
        if let text = value("idle_unload_minutes"), !isOff(text) {
            guard let minutes = Double(text), minutes > 0, minutes <= idleUnloadMaximum else {
                throw CLIError("idle_unload_minutes \"\(text)\" is a number of minutes up to 10080 (7 days), or off")
            }
            settings.idleUnloadMinutes = minutes
        }
        if let text = value("status_messages") {
            guard let flag = parseBool(text) else { throw CLIError("status_messages \"\(text)\" is not true or false") }
            settings.statusMessages = flag
        }
        // A retired `expert_bits` wherever it is still set (a named model's entry included): told, never applied.
        if let text = value("expert_bits") ?? (profile["expert_bits"].flatMap { $0 is NSNull ? nil : "\($0)" }) {
            let from =
                ["flag": "option", "env": "ORIGAMI_EXPERT_BITS", "file": "settings file"][sources["expert_bits"] ?? ""] ?? "model entry"
            settings.retiredExpertBits = "\(text) (\(from))"
        }
        // Sampling: a named model's own values beat the file's keys; a flag or the environment beat both.
        func modelValue(_ key: String) -> String? { value(key) }
        if let text = modelValue("sampling") {
            guard ["recommended", "custom"].contains(text) else { throw CLIError("sampling \"\(text)\" is recommended or custom") }
            settings.samplingCustom = text == "custom"
        }
        for key in SamplingDefaults.keys {
            guard let text = modelValue(key) else { continue }
            settings.samplingValues[key] = try samplingNumber(key, text)
        }
        if let text = modelValue("hot_copy_keep") {
            guard let flag = parseBool(text) else { throw CLIError("hot_copy_keep \"\(text)\" is not true or false") }
            settings.hotCopyKeep = flag
        }
        if let text = modelValue("sampling_override") {
            guard let flag = parseBool(text) else { throw CLIError("sampling_override \"\(text)\" is not true or false") }
            settings.samplingOverride = flag
        }
        if let text = value("commands") {
            guard let flag = parseBool(text) else { throw CLIError("commands \"\(text)\" is not true or false") }
            settings.commands = flag
        }
        if let text = value("context_on_disk"), text != "auto" {
            guard let flag = parseBool(text) else { throw CLIError("context_on_disk \"\(text)\" is not true, false or auto") }
            settings.contextOnDisk = flag
        }
        if let text = value("gpu_keepalive"), text != "auto" {
            guard let flag = parseBool(text) else { throw CLIError("gpu_keepalive \"\(text)\" is not true, false or auto") }
            settings.gpuKeepAlive = flag
        }
        if let text = value("list_models") {
            guard let flag = parseBool(text) else { throw CLIError("list_models \"\(text)\" is not true or false") }
            settings.listModels = flag
        }
        if let text = value("stopped") {
            guard let flag = parseBool(text) else { throw CLIError("stopped \"\(text)\" is not true or false") }
            settings.stopped = flag
        }
        if let text = value("quality") {
            guard let level = QualityDial.level(text) else {
                throw CLIError("quality \"\(text)\" is not \(QualityDial.offeredByModel)")
            }
            settings.quality = level
        }
        // A knob applies on top of `quality` only from a source at least as strong (flag > env > file): `--quality exact`
        // is exact even when the file holds a knob.
        let rank = ["file": 0, "model": 1, "env": 2, "flag": 3]
        let qualityRank = sources["quality"].flatMap { rank[$0] } ?? -1
        for key in qualityKnobs {
            guard let text = value(key) else { continue }
            guard let source = sources[key], rank[source]! >= qualityRank else {
                sources[key] = nil
                continue
            }
            settings.quality = try knob(key, text, on: settings.quality)
        }
        do { settings.quality = try settings.quality.validated() } catch { throw CLIError("quality: \(error)") }
        // Forgiving: a stored level that allow_fast false does not allow serves exact rather than stopping the server.
        if !settings.allowFast && !settings.quality.isExact {
            settings.qualityNote = "quality \(settings.display("quality")) needs allow_fast true; serving exact (allow_fast false)"
            settings.quality = .exact
        }
        settings.sources = sources
        settings.unknownKeys = file.keys.filter { !asked.contains($0) }.sorted()
        return settings
    }

    /// `idle_unload_minutes`' largest value: 7 days.
    static let idleUnloadMaximum = 10_080.0
    /// A custom sampling value from text, in the range a request may send.
    static func samplingNumber(_ key: String, _ text: String) throws -> Double {
        guard let value = Double(text), value.isFinite, SamplingDefaults.valid(key, value) else {
            let range =
                ["temperature": "0 to 100", "top_p": "above 0 to 1", "top_k": "a whole number, 0 (no limit) or more", "min_p": "0 to 1"][
                    key] ?? "-2 to 2"
            throw CLIError("\(key) \"\(text)\" is not \(range)")
        }
        return value
    }

    /// The settings that are secrets: shown as "(set)", never as their text.
    static let secretKeys = ["api_key", "lan_key"]
    /// The quality dial's knobs, each settable alone on top of `quality`'s level.
    static let qualityKnobs = ["experts_per_token", "tail_skip", "tail_skip_misses_only", "late_miss", "confident_above"]
    /// `dial` with one knob set from text, as the settings file, a flag or `origami config set` give it.
    static func knob(_ key: String, _ text: String, on dial: QualityDial) throws -> QualityDial {
        var dial = dial
        switch key {
        case "experts_per_token":
            if ["all", "auto"].contains(text) {
                dial.expertsPerToken = nil
                break
            }
            guard let count = Int(text), (1...10).contains(count) else {
                throw CLIError("experts_per_token \"\(text)\" is not 1 to 10 or all")
            }
            dial.expertsPerToken = count
        case "tail_skip":
            if ["off", "none"].contains(text) {
                dial.tailSkip = nil
                break
            }
            guard let share = Double(text), share > 0, share < 1 else {
                throw CLIError("tail_skip \"\(text)\" is a share of the gate weight between 0 and 1, such as 0.05, or off")
            }
            dial.tailSkip = share
        case "tail_skip_misses_only":
            guard let flag = parseBool(text) else { throw CLIError("tail_skip_misses_only \"\(text)\" is not true or false") }
            dial.tailSkipMissesOnly = flag
        case "confident_above":
            if ["off", "none"].contains(text) {
                dial.confidentAbove = nil
                break
            }
            guard let p = Double(text), p > 0, p < 1 else {
                throw CLIError("confident_above \"\(text)\" is a probability between 0 and 1, such as 0.9, or off")
            }
            dial.confidentAbove = p
        default:
            guard let late = QualityDial.LateMiss(rawValue: text) else {
                throw CLIError("late_miss \"\(text)\" is not wait, skip or substitute")
            }
            dial.lateMiss = late
        }
        return dial
    }

    /// A JSON object file, or empty when there is none. A file that exists but cannot be read or parsed throws, so
    /// the next write never replaces it with a single key.
    static func readConfig(_ url: URL) throws -> [String: Any] {
        guard FileManager.default.fileExists(atPath: url.path) else { return [:] }
        let data: Data
        do { data = try Data(contentsOf: url) } catch {
            throw CLIError("Cannot read \(url.path) (\(error.localizedDescription)); fix its permissions or delete it")
        }
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CLIError("\(url.path) is not valid JSON; fix or delete it")
        }
        return object
    }

    /// Stores settings in the config file (created on first use) in one write; a nil value removes the key. The
    /// read-modify-write holds the file's lock (`StateFile.locked`): the dashboard's Apply and command-line writers
    /// in other processes never lose each other's keys.
    static func write(_ changes: [String: Any?], to url: URL) throws {
        try StateFile.locked(url) {
            var object = try readConfig(url)
            for (key, value) in changes { object[key] = value }
            try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
            // It may hold the API key: readable by this user only.
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        }
    }
    /// `paths` without the data folder's own store (always used) and without repeats.
    static func deduplicated(_ paths: [String], current: Settings) -> [String] {
        var seen: Set<String> = [current.homeStore.standardizedFileURL.path]
        return paths.filter { seen.insert(URL(fileURLWithPath: $0).standardizedFileURL.path).inserted }
    }
    func saveStores() throws {
        var seen: Set<String> = [homeStore.standardizedFileURL.path]
        try Self.write(
            [
                "stores": stores.filter { seen.insert(URL(fileURLWithPath: $0.path).standardizedFileURL.path).inserted }.map {
                    store -> [String: Any] in
                    var entry: [String: Any] = ["path": store.path]
                    if let size = store.size { entry["size"] = size }
                    return entry
                }
            ], to: configFile)
    }

    /// A setting's value to store from text, as `origami config set` and the dashboard take it; nil resets it.
    static func storedValue(_ key: String, _ text: String, current: Settings) throws -> Any? {
        switch key {
        case "context_on_disk", "gpu_keepalive":
            if text == "auto" { return nil }
            guard let flag = parseBool(text) else { throw CLIError("\(key) is true, false or auto") }
            return flag
        case "stores":
            // Keep each listed folder's size limit; the data folder's store (always used) and repeats are dropped.
            return deduplicated(list(text), current: current).map { path -> [String: Any] in
                var entry: [String: Any] = ["path": path]
                if let size = current.stores.first(where: { $0.path == path })?.size { entry["size"] = size }
                return entry
            }
        case "model":
            return ModelRegistry.entry(text, settings: current) != nil || (text.contains("/") && !"/~.".contains(text.first!))
                ? text : expand(text)
        case "data_dir", "context_dir":
            guard !text.trimmingCharacters(in: .whitespaces).isEmpty else {
                throw CLIError("\(key) needs a folder, such as /Volumes/Fast/origami; `origami config unset \(key)` restores the default")
            }
            return expand(text)
        // Off (the default) is stored as no value; anything else is a path.
        case "speculation", "request_log": return isOff(text) ? nil : expand(text)
        case "hot_copy_dir":
            guard !isOff(text) else { throw CLIError(hotCopyDirOff) }
            return text == "auto" ? nil : expand(text)
        case "hot_copy_more":
            let places = try copyPlaces(text)
            if let first = current.hotCopyDir, places.contains(where: { $0.folder == first }) {
                throw CLIError("hot_copy_more: \(first.path) is hot_copy_dir already")
            }
            return copyPlacesText(places)
        case "models_dirs":
            let folders = list(text)
            return folders.isEmpty ? nil : String(data: try JSONSerialization.data(withJSONObject: folders), encoding: .utf8)
        case "api_key":
            if isOff(text) { return nil }
            guard isAPIKey(text) else {
                throw CLIError("api_key is 16 to 256 characters without spaces, such as `openssl rand -hex 16` prints, or off")
            }
            return text
        case "listen":
            guard ["local", "lan"].contains(text.lowercased()) else {
                throw CLIError("listen is local (this Mac only) or lan (also other devices on this network)")
            }
            return text.lowercased() == "local" ? nil : "lan"
        case "trusted_proxy":
            guard isOff(text) || text.lowercased() == "loopback" else {
                throw CLIError("trusted_proxy is loopback (a TLS proxy on this Mac names each client in X-Forwarded-For) or off")
            }
            return isOff(text) ? nil : "loopback"
        // on: keep the key there is, else make one; off: no key for other devices.
        case "lan_key":
            if isOff(text) { return "off" }
            if text.lowercased() == "on" { return current.lanKey ?? newKey() }
            guard isAPIKey(text) else {
                throw CLIError("lan_key is 16 to 256 characters without spaces, on (make one) or off (none: less safe)")
            }
            return text
        case "memory_limit", "expert_memory", "hot_copy":
            if text == "auto" || (key == "hot_copy" && text == "balanced") { return nil }
            guard parseBytes(text) != nil || (key == "hot_copy" && Self.hotCopyPreset(text) != nil) else {
                throw CLIError("\(key) is a size such as 20GB\(key == "hot_copy" ? ", off, save-space, balanced, fastest" : "") or auto")
            }
            if key == "memory_limit", let bytes = parseBytes(text) {
                let maximum = WiredMemoryPolicy.maximumBudgetBytes(physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)
                guard bytes > 0, bytes <= maximum else {
                    throw CLIError(
                        "memory_limit must be positive and at most \(OrigamiPlan.gigabytes(maximum)) on this Mac, leaving room for macOS and other apps"
                    )
                }
            }
            return text
        case "context":
            guard let tokens = parseContext(text), tokens <= OrigamiPlan.maximumContext else {
                throw CLIError("context is a token count up to 262144, such as 8k, 32k, 100000 or 256k")
            }
            return tokens % 1024 == 0 ? "\(tokens / 1024)k" : "\(tokens)"
        case "port":
            guard let port = Int(text), (1...65535).contains(port) else { throw CLIError("\(text) is not a port number") }
            return port
        case "reasoning_effort":
            guard ServingDefaults.efforts.contains(text) else {
                throw CLIError("reasoning_effort is one of \(ServingDefaults.efforts.joined(separator: ", "))")
            }
            return text
        case "max_thinking_tokens":
            guard let tokens = Int(text), tokens >= 1 else { throw CLIError("max_thinking_tokens is a positive number") }
            return tokens
        case "idle_unload_minutes":
            if isOff(text) { return nil }
            guard let minutes = Double(text), minutes > 0, minutes <= idleUnloadMaximum else {
                throw CLIError("idle_unload_minutes is a number of minutes up to 10080 (7 days), or off")
            }
            return minutes
        case "expert_bits": throw CLIError("expert_bits " + expertBitsRetired)
        case "prompt_cache_entries":
            guard let count = Int(text), PromptCache.entryRange.contains(count) else {
                throw CLIError("prompt_cache_entries is \(PromptCache.entryRange.lowerBound) to \(PromptCache.entryRange.upperBound)")
            }
            return count
        case "sampling":
            guard ["recommended", "custom"].contains(text) else { throw CLIError("sampling is recommended (the model's own) or custom") }
            return text == "recommended" ? nil : text
        case "temperature", "top_p", "top_k", "min_p", "presence_penalty":
            let value = try samplingNumber(key, text)
            return key == "top_k" ? Int(value) as Any : value
        case "quality":
            // Stored under its own name (`faster` as `fast`).
            guard let level = QualityDial.level(text) else {
                throw CLIError("quality is \(QualityDial.offeredByModel)")
            }
            return level.name
        case "experts_per_token", "tail_skip", "late_miss", "confident_above":
            _ = try knob(key, text, on: current.quality)
            return text
        default:
            guard let flag = parseBool(text) else { throw CLIError("\(key) is true or false") }
            return flag
        }
    }

    /// An absolute path without `.` or `..` components (the engine refuses a prompt-cache folder with `..`); lexical
    /// only, so a drive that does not answer is never touched.
    static func expand(_ path: String) -> String {
        let expanded = (path as NSString).expandingTildeInPath
        let absolute = expanded.hasPrefix("/") ? expanded : FileManager.default.currentDirectoryPath + "/" + expanded
        return URL(fileURLWithPath: absolute, isDirectory: false).standardized.path
    }
    /// JSON folder arrays preserve commas in names; legacy comma-separated lists remain supported.
    static func list(_ text: String) -> [String] {
        if let data = text.data(using: .utf8), let paths = try? JSONDecoder().decode([String].self, from: data) { return paths.map(expand) }
        return text.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }.map(expand)
    }
    /// `hot_copy_more` from its text: `folder=size` pairs, comma-separated (a size as `hot_copy` takes it, `auto` when
    /// none), or the JSON list it is stored as (`[{"folder": …, "size": bytes or "auto"}]`); off or empty: none. Two
    /// places on one connected drive are refused: they would share its bandwidth and its reserve.
    static func copyPlaces(_ text: String) throws -> [CopyPlace] {
        let usage = "hot_copy_more is folder=size pairs, comma-separated, such as /Volumes/Fast/Origami/hot-copy=200GB, or off"
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        if isOff(trimmed) { return [] }
        var entries: [(folder: String, size: String)] = []
        if trimmed.hasPrefix("[") {
            guard let list = (try? JSONSerialization.jsonObject(with: Data(trimmed.utf8))) as? [[String: Any]] else {
                throw CLIError(usage)
            }
            for entry in list {
                guard let folder = entry["folder"] as? String else { throw CLIError(usage) }
                let size = entry["size"].map { ($0 as? String) ?? (($0 as? NSNumber).map { "\($0.uint64Value)B" } ?? "") } ?? "auto"
                entries.append((folder, size))
            }
        } else {
            for item in trimmed.split(separator: ",").map({ $0.trimmingCharacters(in: .whitespaces) }) where !item.isEmpty {
                let cut = item.lastIndex(of: "=")
                entries.append(cut.map { (String(item[..<$0]), String(item[item.index(after: $0)...])) } ?? (item, "auto"))
            }
        }
        var out: [CopyPlace] = []
        for (folder, size) in entries {
            guard folder.hasPrefix("/") || folder.hasPrefix("~") else { throw CLIError("hot_copy_more needs full paths (not \(folder))") }
            let url = URL(fileURLWithPath: expand(folder), isDirectory: true)
            let lower = size.lowercased().trimmingCharacters(in: .whitespaces)
            let bytes: UInt64?
            if ["auto", "balanced", ""].contains(lower) {
                bytes = nil
            } else if lower.hasSuffix("b"), let exact = UInt64(lower.dropLast()) {
                bytes = exact
            } else if let parsed = hotCopyPreset(lower) ?? parseBytes(lower) {
                bytes = parsed
            } else {
                throw CLIError("hot_copy_more: \"\(size)\" is a size such as 200GB, off, auto, or save-space, balanced, fastest")
            }
            guard !out.contains(where: { $0.folder == url }) else { throw CLIError("hot_copy_more names \(url.path) twice") }
            let volume = Stores.volume(url)
            if volume.mounted,
                let other = out.first(where: { Stores.volume($0.folder).mounted && Stores.volume($0.folder).disk == volume.disk })
            {
                throw CLIError(
                    "hot_copy_more: \(url.path) and \(other.folder.path) are on the same drive; one place per drive (give it the size of both)"
                )
            }
            out.append(CopyPlace(folder: url, size: bytes))
        }
        return out
    }
    /// The text `hot_copy_more` is stored as: a JSON list with each size in bytes, or "auto"; nil for none.
    static func copyPlacesText(_ places: [CopyPlace]) -> String? {
        guard !places.isEmpty else { return nil }
        let list: [[String: Any]] = places.map { ["folder": $0.folder.path, "size": $0.size.map { $0 as Any } ?? "auto"] }
        return (try? JSONSerialization.data(withJSONObject: list, options: [.sortedKeys, .withoutEscapingSlashes])).flatMap {
            String(data: $0, encoding: .utf8)
        }
    }
    /// `hot_copy_dir off` would read as automatic: the copy is turned off by `hot_copy`.
    static let hotCopyDirOff = "hot_copy_dir is a folder or auto; `origami config set hot_copy off` turns the hot copy off"
    static func isOff(_ text: String) -> Bool { ["off", "false", "none", ""].contains(text.lowercased()) }
    /// 32 hex characters from the system's random source, for `lan_key`.
    static func newKey() -> String {
        var bytes = [UInt8](repeating: 0, count: 16)
        guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else { fatalError("SecRandomCopyBytes failed") }
        return bytes.map { String(format: "%02x", $0) }.joined()
    }
    /// What else to save with `changes` so other devices get a key: `listen lan` without `api_key` and without
    /// `lan_key` makes one (programs on this Mac keep working without it).
    func lanKeyFor(_ changes: [String: Any?]) -> [String: Any?] {
        let lan = changes.keys.contains("listen") ? (changes["listen"] ?? nil) as? String == "lan" : listenLAN
        let key = changes.keys.contains("lan_key") ? (changes["lan_key"] ?? nil) as? String : lanOpen ? "off" : lanKey
        let api = changes.keys.contains("api_key") ? (changes["api_key"] ?? nil) as? String : apiKey
        return lan && key == nil && api == nil ? ["lan_key": Self.newKey()] : [:]
    }
    /// An API key the HTTP headers carry unchanged: 16 to 256 visible ASCII characters.
    static func isAPIKey(_ text: String) -> Bool {
        (16...256).contains(text.utf8.count) && text.utf8.allSatisfy { (0x21...0x7e).contains($0) }
    }
    /// "20GB", "20G", "500MB", "20" (gigabytes) to bytes, decimal units as macOS shows them; nil for anything else,
    /// including negative, nan, inf and sizes past 2^64, which would trap in the conversion.
    static func parseBytes(_ text: String) -> UInt64? {
        let lower = text.lowercased().replacingOccurrences(of: " ", with: "")
        let (suffix, scale) =
            [("tb", 1e12), ("t", 1e12), ("gb", 1e9), ("g", 1e9), ("mb", 1e6), ("m", 1e6)].first { lower.hasSuffix($0.0) } ?? ("", 1e9)
        guard let value = Double(lower.dropLast(suffix.count)).map({ $0 * scale }), value.isFinite, value >= 0, value < 0x1p64 else {
            return nil
        }
        return UInt64(value)
    }
    /// "32k", "100000" to tokens; nil for anything else (bounded before the multiply, so no count can overflow).
    static func parseContext(_ text: String) -> Int? {
        let lower = text.lowercased()
        let tokens = lower.hasSuffix("k") ? Int(lower.dropLast()).flatMap { (1...(1 << 20)).contains($0) ? $0 * 1024 : nil } : Int(lower)
        return tokens.flatMap { $0 > 0 ? $0 : nil }
    }
    static func parseBool(_ text: String) -> Bool? {
        switch text.lowercased() {
        case "true", "yes", "1", "on": return true
        case "false", "no", "0", "off": return false
        default: return nil
        }
    }
}

/// A failure `main` prints as `origami: <description>`: exit status 2 for a usage error, 1 for anything else.
struct CLIError: Error, CustomStringConvertible {
    let description: String
    /// 1 for a failure, 2 for a usage error.
    var exitCode: Int32 = 1
    init(_ description: String) { self.description = description }
    static func usage(_ description: String) -> Self {
        var error = Self(description)
        error.exitCode = 2
        return error
    }
}
