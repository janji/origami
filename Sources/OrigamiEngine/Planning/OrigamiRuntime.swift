import CryptoKit
import Foundation
import Metal

/// A serving backend that answers at once and forwards to the real engine once it has
/// loaded, so wrappers (llama-swap, launchd) see the port listening in a second.
/// Completions sent while loading wait in the server's queue.
public final class DeferredBackend: ServingBackend, @unchecked Sendable {
    private let lock = NSLock()
    private var inner: NativeServingBackend?
    private var failure: String?
    private var loadStage: String?
    private let version: String, context: Int
    public init(version: String, context: Int) {
        self.version = version
        self.context = context
    }

    public func ready(_ backend: NativeServingBackend) {
        lock.lock()
        inner = backend
        lock.unlock()
    }
    /// What the load is doing now when it may take long (the saved prompts' canary), for `/health`'s `note`.
    public var stage: String? {
        get { lock.withLock { loadStage } }
        set { lock.withLock { loadStage = newValue } }
    }
    public func failed(_ reason: String) {
        lock.lock()
        failure = reason
        lock.unlock()
    }
    public var backend: NativeServingBackend? {
        lock.lock()
        defer { lock.unlock() }
        return inner
    }

    public func status() -> ServingStatus {
        lock.lock()
        let inner = self.inner
        let failure = self.failure
        lock.unlock()
        if let inner { return inner.status() }
        var status = ServingStatus(version: version, contextTier: context, ready: false)
        status.failure = failure
        status.degraded = stage  // shown as `note` while loading
        return status
    }
    public func prepare(_ request: ServingRequest, cancellation: GenerationControl) throws -> any PreparedGeneration {
        lock.lock()
        let backend = inner
        let failure = self.failure
        lock.unlock()
        guard let backend else {
            throw ServingFailure(
                failure.map { "The model failed to load: \($0)" } ?? "The model is still loading", status: 503, type: "engine_unavailable")
        }
        return try backend.prepare(request, cancellation: cancellation)
    }
}

/// Builds the serving engine for one model on this Mac: weights store, memory plan,
/// prompt cache, executor, tokenizer and memory governor.
public enum OrigamiRuntime {
    /// What `load` built. Unchecked Sendable: every member is either internally locked or immutable after load.
    public struct Loaded: @unchecked Sendable {
        public let backend: NativeServingBackend
        public let governor: MemoryGovernor
        public let plan: OrigamiPlan
        public let promptCache: PromptCache
        /// The chat template's sha256 and startup probe (for /v1/status).
        public let template: [String: Any]
    }

    public static func load(
        model: ModelSource, host: OrigamiHost, storage: OrigamiStorage, context: Int, memoryLimit: UInt64?, expertMemory: UInt64? = nil,
        tier: Int? = nil, inMemory: Bool = false,
        allowFast: Bool, promptCacheRoot: URL, version: String, routingFile: URL? = nil,
        mtpWeights: URL? = nil, promptCacheEntries: Int = PromptCache.defaultEntries, gpuKeepAlive: Bool? = nil,
        qwenPrefillScope: Int? = nil,
        memoryHistory: MemoryHistory? = nil, learnsMemory: Bool = false,
        stage: @escaping @Sendable (String?) -> Void = { _ in }, log: @escaping @Sendable (String) -> Void
    ) async throws -> Loaded {
        let initialSwapouts = swapoutBytes()
        ModelLayout.use(model, serving: true)
        let store = try PackStore(model: model, pack: storage.packFolder, copies: storage.copyFolders, log: log)
        var storage = storage
        if storage.packFolder != nil && !store.packedAvailable { storage.packFolder = nil }
        // Diagnostic execution switches, e.g. ORIGAMI_EXECUTION='{"prefill_double_buffer":false}';
        // the text enters the prompt-cache epoch, so overridden runs never share saved states with plain ones.
        let overridesText = ProcessInfo.processInfo.environment["ORIGAMI_EXECUTION"] ?? ""
        guard
            var overrides = try overridesText.isEmpty ? [:] : JSONSerialization.jsonObject(with: Data(overridesText.utf8)) as? [String: Any]
        else {
            throw EngineError(
                "ORIGAMI_EXECUTION must be a JSON object of execution switches, such as {\"prefill_double_buffer\":false}",
                code: "invalid_configuration")
        }
        if !overrides.isEmpty { log("execution overrides: \(overridesText)") }
        // The `gpu_keepalive` setting (nil: the plan's choice) under ORIGAMI_EXECUTION; it only schedules a spin command,
        // so it stays out of the epoch and saved prompts stay shared.
        if let gpuKeepAlive, overrides["gpu_keepalive"] == nil { overrides["gpu_keepalive"] = gpuKeepAlive }
        let plan = try OrigamiPlan.make(
            model: model, host: host, context: context, memoryLimit: memoryLimit, expertMemory: expertMemory, allowFast: allowFast,
            storage: storage, promptCache: promptCacheRoot, overrides: overrides, tier: tier, inMemory: inMemory,
            warn: log)
        for note in plan.notes { log(note) }
        let runtime = plan.configuration
        // Group fewer tiles within the already reserved R8192 allocation. Explicit diagnostic geometry wins.
        let scopeSet = (overrides["prefill"] as? [String: Any])?["read_scope"] != nil
        let prefillScope =
            runtime.computeTile == 256 && !scopeSet
            ? min(runtime.readScope, max(256, (qwenPrefillScope ?? runtime.readScope) / 256 * 256)) : runtime.readScope
        let ledger = try runtime.makeLedger()
        let tokenizer = try await TokenizerStack(modelDirectory: model.directory)

        // Everything a saved state depends on: the engine's arithmetic (not its build: builds that compute the same bits
        // share saved prompts, `PromptCanary`), weights, tokenizer, template and settings.
        let config = try Data(contentsOf: model.directory.appendingPathComponent("config.json"))
        let template = (try? Data(contentsOf: model.directory.appendingPathComponent("chat_template.jinja"))) ?? Data()
        let arithmetic = ArithmeticDigest.forQwen(ggufExperts: model.ggufExperts != nil)
        func epoch(_ fingerprint: String, _ digest: String) -> String {
            PromptCanary.epoch(
                fingerprint + tokenizer.identity + "\(runtime.contextLimit)" + String(describing: runtime.experts.mode) + overridesText
                    + "|prefill-scope:\(prefillScope)", arithmetic: digest)
        }
        let key = epoch(model.fingerprint, arithmetic.value)
        let root = promptCacheRoot.appendingPathComponent(String(key.prefix(16)), isDirectory: true)
        // The folder's saves may have been written under an earlier key of the same bits (`PromptCanary.adopt`): 3.1.8.3's
        // single digest while this engine's files are unchanged since, and the fingerprints the model had before refreshes
        // that read its bytes back unchanged (`ModelRefresh.earlierFingerprints`), latest first.
        let fingerprints = [model.fingerprint] + ModelRefresh.earlierFingerprints(promptCacheRoot).reversed()
        let earlier = fingerprints.flatMap { fingerprint in
            ([arithmetic.value] + [arithmetic.before].compactMap { $0 }).map { (epoch: epoch(fingerprint, $0), model: fingerprint) }
        }
        let written = PromptCanary.adopt(root, epoch: key, model: model.fingerprint, arithmetic: arithmetic.value, from: earlier)
        let identity = try CheckpointIdentity(
            modelSHA256: written.model, tokenizerSHA256: tokenizer.identity,
            templateSHA256: hex(SHA256.hash(data: template)), quantizationSHA256: hex(SHA256.hash(data: config)),
            arithmeticEpoch: "origami-" + written.epoch)
        let geometry = try ConversationGeometry(context: runtime.contextLimit, fileBacked: runtime.fileBackedContext)
        // One folder per epoch, shared by the builds whose canary matches (checked below, before any request).
        let promptCache = try PromptCache(
            root: root, epoch: written.epoch,
            // Plus four system snapshots. No aggregate quota: PromptCache.makeRoom keeps 10 GB free on the drive.
            entries: PromptCache.regularEntries(context: runtime.contextLimit, entries: promptCacheEntries),
            entryBytes: runtime.prefixDiskBytes, ledger: ledger, capacity: runtime.contextLimit,
            // A context kept in a file grows on the same drive: saves leave it room too.
            minimumFreeBytes: 10_000_000_000 + (geometry.fileBacked ? ConversationGeometry.fileDiskBytes : 0),
            build: PromptCanary.build, arithmetic: arithmetic.value)
        promptCache.log = log
        let prefix = try PrefixReuseController(
            identity: identity, ledger: ledger, memoryBytes: runtime.prefixMemoryBytes,
            promptCache: promptCache, conversationContext: geometry.context, conversationFileBacked: geometry.fileBacked,
            frontierDiskBytes: runtime.conversationFrontierDiskBytes,
            frontierAsyncWrites: runtime.frontierAsyncWrites, frontierWriteStagingSlots: runtime.frontierWriteStagingSlots)

        let executor = try runtime.makeExecutor(store: store, ledger: ledger)
        let driver = GenerationDriver(
            executor: executor, tokenizer: tokenizer, prefixReuse: prefix,
            keepAlive: try runtime.makeKeepAlive(ledger: ledger))
        driver.keepAliveByDefault = runtime.gpuKeepAlive
        if let mtpWeights {
            // The head's experts are 4-bit group-64 records like the checkpoint's own; other layouts decode plainly.
            if ModelLayout.experts == AffineFormat(bits: 4, group: 64), ModelLayout.ggufExperts == nil,
                let device = MTLCreateSystemDefaultDevice()
            {
                // The head is optional: a file that cannot be read turns speculation off instead of stopping the server.
                do {
                    driver.speculation = try MTPHead(file: mtpWeights, device: device, ledger: ledger)
                    log("speculation: MTP head from \(mtpWeights.lastPathComponent); plain decode wherever it measures slower")
                } catch {
                    log(
                        "speculation: off (\(mtpWeights.path) could not be used: \(error); `origami config set speculation off` silences this)"
                    )
                }
            } else {
                log(
                    "speculation: off (the MTP head needs the 4-bit group-64 expert layout"
                        + (ModelLayout.ggufExperts == nil ? ")" : "; this model's experts are read from its GGUF)"))
            }
        }
        let backend = NativeServingBackend(
            driver: driver, version: version, computeTile: runtime.computeTile,
            readScope: runtime.readScope, prefillReadScope: prefillScope)
        backend.expectsPack = storage.packFolder != nil
        backend.servedContext = min(context, runtime.contextLimit)
        if model.tensors.keys.contains(where: { $0.hasPrefix("vision_tower.") }) { backend.visionModel = model }
        backend.log = log
        let governor = MemoryGovernor(
            ledger: ledger, memory: plan.memory, trim: { backend.trimToMemoryBudget() }, log: log,
            history: memoryHistory, learns: learnsMemory, initialSwapouts: initialSwapouts,
            grow: { backend.growToMemoryBudget() })
        governor.start()
        try backend.prepareReadiness()
        let readiness = backend.readinessPreparation
        if let seconds = readiness["seconds"] as? Double, let layers = readiness["fixed_layers"] as? Int {
            log(
                String(
                    format: "startup preparation: %d fixed layers in %.3f s (%@)", layers, seconds,
                    readiness["reason"] as? String ?? "unknown"))
        }
        // A build the folder's canary does not list yet runs the probe now, before routing counts are kept and any request.
        if promptCache.savesOff == PromptCache.checking { stage(PromptCache.checking) }
        promptCache.settle { try backend.canary() }
        stage(nil)
        if let off = promptCache.savesOff { log("prompt cache: off (\(off))") }
        if let note = promptCache.savesNote { log("prompt cache: \(note)") }
        if let note = promptCache.unusableNote { log("saved prompts: \(note)") }
        if let note = promptCache.canaryNote { log("saved prompts: \(note)") }
        if let routingFile {
            executor.routingProfile = RoutingProfile.load(routingFile) ?? RoutingProfile()
            // Only the server that owns the data folder writes its counts back.
            backend.routingFile = promptCache.shared ? nil : routingFile
        }
        let templateSHA = hex(SHA256.hash(data: template))
        let probe = tokenizer.templateCheck()
        if probe == nil { log("the chat template (sha256 \(templateSHA.prefix(12))) could not render a test conversation") }
        if probe?.thinkingSwitch == false {
            log(
                "the chat template (sha256 \(templateSHA.prefix(12))) ignores enable_thinking: reasoning_effort none cannot turn thinking off"
            )
        }
        return Loaded(
            backend: backend, governor: governor, plan: plan, promptCache: promptCache,
            template: [
                "sha256": templateSHA, "thinking_switch": probe?.thinkingSwitch ?? false,
                "follow_up_extends": probe?.followUpExtends ?? false,
            ])
    }

    /// The running binary's digest: the builds a saved-prompt folder's canary lists (`PromptCanary`).
    package static func executableDigest() throws -> String {
        guard let url = Bundle.main.executableURL else { return "unknown-build" }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        while let data = try handle.read(upToCount: 4 * 1024 * 1024), !data.isEmpty { hash.update(data: data) }
        return hex(hash.finalize())
    }
    package static func hex<D: Sequence>(_ digest: D) -> String where D.Element == UInt8 {
        digest.map { String(format: "%02x", $0) }.joined()
    }
}
