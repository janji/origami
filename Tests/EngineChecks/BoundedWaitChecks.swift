import CryptoKit
import Foundation
import Metal
import OrigamiEngine

/// ROADMAP R2, nothing waits forever: bounded command-buffer waits (`GPUWait`), bounded demand reads
/// (`ExpertReadStall`), the request time limit (`origami_time_limit`) and the front's restart of a dead model process,
/// on fakes and on the synthetic family models.
func boundedWaitChecks() -> [(String, () throws -> Void)] {
    [
        ("gpu_wait_bounded", gpuWaitBounded),
        ("gpu_watchdog_earlier_deadline", gpuWatchdogEarlierDeadline),
        ("expert_read_stall_bounded", expertReadStallBounded),
        ("dead_model_process_restarts", deadModelProcessRestarts),
        ("prompt_cache_save_refused_after_lost_gpu", promptCacheSaveRefusedAfterLostGPU),
        ("synthetic_gpu_hang_recovers", { for family in [ModelFamily.glm, .deepseek] { try syntheticGPUHang(family) } }),
        ("synthetic_read_stall_recovers", { try syntheticReadStall(.glm) }),
        ("synthetic_guess_stall_bounded", syntheticGuessStall),
        ("synthetic_background_read_stall_bounded", syntheticBackgroundReadStall),
        ("qwen_row_prefetch_stall_bounded", qwenRowPrefetchStallBounded),
        ("synthetic_time_limit_keeps_conversation", { for family in [ModelFamily.glm, .deepseek] { try syntheticTimeLimit(family) } }),
    ]
}

private func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("bounded waits: " + what) } }

/// Runs `body` with short bounds, and leaves the defaults and a usable device behind.
private func withShortBounds<T>(gpu: Double = 1, read: Double = 60, _ body: () throws -> T) rethrows -> T {
    let saved = (GPUWait.floor, GPUWait.settle, ExpertReadStall.floor, ExpertReadStall.multiple)
    GPUWait.floor = gpu
    GPUWait.settle = 2
    ExpertReadStall.floor = read
    defer {
        (GPUWait.floor, GPUWait.settle, ExpertReadStall.floor, ExpertReadStall.multiple) = saved
        GPUWait.inject([])
        GPUWait.recover()
        GPUWait.onLost = nil
    }
    return try body()
}

private final class Flag: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    func raise() { lock.withLock { count += 1 } }
    var raised: Int { lock.withLock { count } }
}

/// The watchdog sleeping toward a long wait's deadline: a shorter wait opened after that one closed, or
/// beside it, is still taken for lost at its own deadline, not at the older, later one.
private func gpuWatchdogEarlierDeadline() throws {
    try withShortBounds {
        let lost = Flag()
        GPUWait.onLost = { _ in lost.raise() }
        let long = GPUWait.watchdog.register(deadline: Date() + 30, message: "long")
        usleep(50_000)  // the watchdog now sleeps toward the long deadline
        _ = GPUWait.watchdog.close(long)
        let started = Date()
        let short = GPUWait.watchdog.register(deadline: started + 0.2, message: "short")
        while lost.raised == 0 && Date().timeIntervalSince(started) < 10 { usleep(5_000) }
        let seconds = Date().timeIntervalSince(started)
        try require(GPUWait.watchdog.close(short) && lost.raised == 1 && seconds < 2, "a 0.2 s wait was taken for lost after \(seconds) s")
        GPUWait.recover()
        // Beside a longer open wait as well.
        let open = GPUWait.watchdog.register(deadline: Date() + 30, message: "open")
        usleep(50_000)
        let again = Date()
        let beside = GPUWait.watchdog.register(deadline: again + 0.2, message: "beside")
        while lost.raised == 1 && Date().timeIntervalSince(again) < 10 { usleep(5_000) }
        let besideSeconds = Date().timeIntervalSince(again)
        try require(
            GPUWait.watchdog.close(beside) && !GPUWait.watchdog.close(open) && lost.raised == 2 && besideSeconds < 2,
            "a 0.2 s wait beside a 30 s one was taken for lost after \(besideSeconds) s")
    }
}

/// A watched buffer seen ended within its bound returns (a slow but healthy GPU too); one seen late but within `settle`
/// more is `gpu_stalled` (retried, the budget halved); one never seen is `gpu_lost`, after which every wait fails at once
/// until the process restarts. The bound follows the work and the budget.
private func gpuWaitBounded() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
        let buffer = device.makeBuffer(length: 1 << 16, options: .storageModeShared)
    else { throw FixtureUnavailable(description: "no Metal device") }
    try withShortBounds {
        let lost = Flag()
        GPUWait.onLost = { _ in lost.raise() }
        let budget = CommandBudget()
        func run(_ delay: Double??) -> (seconds: Double, error: Error?, status: MTLCommandBufferStatus) {
            if let delay { GPUWait.inject([delay]) }
            let command = queue.makeCommandBuffer()!
            let blit = command.makeBlitCommandEncoder()!
            blit.fill(buffer: buffer, range: 0..<buffer.length, value: 7)
            blit.endEncoding()
            command.commit()
            let started = Date()
            do { try command.waitBounded(budget: budget) } catch {
                return (Date().timeIntervalSince(started), error, command.status)
            }
            return (Date().timeIntervalSince(started), nil, command.status)
        }
        func code(_ error: Error?) -> String? { (error as? EngineError)?.code }
        let plain = run(nil)
        try require(plain.error == nil && plain.status == .completed && plain.seconds < 0.3, "a plain wait: \(plain)")
        let slow = run(.some(0.15))
        try require(slow.error == nil && slow.seconds >= 0.15 && budget.limit == CommandBudget.work, "a slow healthy GPU tripped: \(slow)")
        let late = run(.some(1.5))
        try require(
            code(late.error) == "gpu_stalled" && late.seconds >= 1.5 && late.seconds < 3 && late.status == .completed,
            "a late completion: \(late)")
        try require(budget.limit == CommandBudget.work / 2 && GPURetry.isInterruption(late.error!), "a stall is not retried as an abort")
        try require(lost.raised == 0 && !GPUWait.lost, "a stall took the GPU for lost")
        let never = run(.some(nil))
        try require(code(never.error) == "gpu_lost" && never.seconds >= 3 && never.seconds < 4, "a lost completion: \(never)")
        try require(lost.raised == 1 && GPUWait.lost && !GPURetry.isInterruption(never.error!), "the lost GPU was not reported once")
        let after = run(nil)
        try require(code(after.error) == "gpu_lost" && after.seconds < 0.1 && lost.raised == 1, "a wait after the loss: \(after)")
        try require(
            ServingFailure.from(never.error!).status == 503 && ServingFailure.from(late.error!).status == 503, "the client's status")
        GPUWait.recover()
        try require(run(nil).error == nil, "a recovered device still fails")
        // The bound: `multiple` times the budget's expected time, at least `floor`; a halved budget doubles it.
        GPUWait.floor = 10
        let fresh = CommandBudget()
        try require(GPUWait.bound(work: 0, budget: nil) == 10, "the floor")
        try require(
            abs(GPUWait.bound(work: CommandBudget.work * 100, budget: fresh) - 20 * 100 * CommandBudget.seconds) < 1e-9, "the multiple")
        fresh.interrupted()
        try require(
            abs(GPUWait.bound(work: CommandBudget.work * 100, budget: fresh) - 2 * 20 * 100 * CommandBudget.seconds) < 1e-9, "the budget")
    }
}

/// A demand batch with a read that hangs throws `drive_stalled` at its bound without waiting for it, and frees nothing
/// (`stalled` runs) until that read has returned.
private func expertReadStallBounded() throws {
    let ledger = try MemoryLedger(ceiling: 1 << 30, transientMargin: 1, availableFloor: 1)
    let scheduler = try FamilyExperts.makeReadScheduler(ledger: ledger, jobs: 8)
    let gate = DispatchSemaphore(value: 0)
    let freed = Flag()
    var opened = false
    // Whatever fails below, the hung read returns: a reader left blocked would hang the process at teardown.
    defer { if !opened { gate.signal() } }
    let started = Date()
    do {
        try FamilyExperts.perform(3, on: scheduler, bytes: { _ in 1 << 20 }, spin: nil, stall: 0.3, stalled: { freed.raise() }) { i in
            if i == 1 { gate.wait() }
        }
        throw EngineError("bounded waits: a hung read was not bounded")
    } catch let error as EngineError where error.code == "drive_stalled" {}
    let seconds = Date().timeIntervalSince(started)
    try require(seconds >= 0.3 && seconds < 5, "the stall took \(seconds) s")
    usleep(100_000)
    try require(freed.raised == 0, "the stalled batch was released while its read hung")
    opened = true
    gate.signal()
    let until = Date().addingTimeInterval(5)
    while freed.raised == 0 && Date() < until { usleep(1_000) }
    try require(freed.raised == 1, "the stalled batch was not released once its read returned")
    // The message only: `ServingFailure.from` would log it, which reads like a real 60 s wait.
    try require(ExpertReadStall.error(bytes: 1 << 20, seconds: 60).description.contains("stopped answering"), "the message")
    try require(ExpertReadStall.bound(bytes: 0, bytesPerSecond: 1e9) == ExpertReadStall.floor, "the read floor")
    try require(ExpertReadStall.bound(bytes: 100 << 30, bytesPerSecond: 1e9) > ExpertReadStall.floor, "the read multiple")
}

/// A prompt-cache save whose state was read after the GPU was taken for lost is not published: refused before it writes,
/// or, when the GPU is lost while it writes, its files removed and no entry made. Once the device is usable it saves.
private func promptCacheSaveRefusedAfterLostGPU() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-lost-save-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let ledger = try MemoryLedger(ceiling: 1 << 30, transientMargin: 1, availableFloor: 1)
    let cache = try PromptCache(
        root: folder, epoch: "lost", entries: 3, entryBytes: PromptCache.entryDiskBytes(context: 8_192) * 2, ledger: ledger,
        capacity: 8_192, minimumFreeBytes: 0)
    let tokens = Array(0..<600)
    let ids = tokens.map(Int64.init)[...]
    var written: URL?
    func save(losing: Bool) -> String? {
        do {
            try cache.writeState(tokens: tokens, bytes: 1 << 20) { folder in
                written = folder
                try Data("published".utf8).write(to: FamilyStateFile.headerURL(folder))
                if losing { _ = GPUWait.declareLost("a check's lost GPU") }
            }
            return nil
        } catch { return (error as? EngineError)?.code ?? "\(error)" }
    }
    try withShortBounds {
        _ = GPUWait.declareLost("a check's lost GPU")
        try require(save(losing: false) == "gpu_lost" && written == nil && !cache.contains(ids), "a save after the loss was written")
        GPUWait.recover()
        try require(save(losing: true) == "gpu_lost" && !cache.contains(ids), "a save the loss interrupted was published")
        try require(
            written.map { !FileManager.default.fileExists(atPath: FamilyStateFile.headerURL($0).path) } == true,
            "a save the loss interrupted left its files")
        GPUWait.recover()
        try require(save(losing: false) == nil && cache.contains(ids), "a save on a usable device failed")
    }
}

/// The front's scheduler starts a model process again when the running one has exited (as a lost GPU makes it).
private func deadModelProcessRestarts() throws {
    final class Child {
        let id: String
        var exited = false
        init(_ id: String) { self.id = id }
    }
    var started = 0
    let scheduler = ModelScheduler<Child>(
        start: {
            started += 1
            return Child($0)
        }, stop: { _ in }, alive: { !$0.exited })
    let first = try scheduler.acquire("a", fallback: "a").child
    scheduler.release()
    first.exited = true
    let second = try scheduler.acquire("a", fallback: "a").child
    scheduler.release()
    try require(second !== first && started == 2, "a dead model process was not started again")
}

// MARK: the synthetic family models

func sha256(_ model: FamilyModel) -> String {
    var whole = SHA256()
    for part in model.stateParts(at: model.position) {
        whole.update(bufferPointer: UnsafeRawBufferPointer(start: part.buffer.contents(), count: part.bytes))
    }
    return whole.finalize().map { String(format: "%02x", $0) }.joined()
}

/// A synthetic model behind a serving backend with a prompt cache, as `origami serve` runs it.
final class SyntheticServer {
    let model: FamilyModel
    let runner: FamilyRunner
    let backend: FamilyServingBackend
    init(_ weights: FamilyWeights, tokenizer: TokenizerStack, folder: URL) throws {
        let routed = weights.gguf.tensors.values.filter { $0.name.contains("_exps.") }.reduce(0) { $0 + Int($1.length) }
        model = try makeFamilyModel(weights, expertBudget: UInt64(routed * 2 / 5), context: 8_192) { _ in }
        runner = model as! FamilyRunner
        let ledger = try MemoryLedger(ceiling: 1 << 30, transientMargin: 1, availableFloor: 1)
        let cache = try PromptCache(
            root: folder.appendingPathComponent("cache-\(UUID().uuidString)"), epoch: "synthetic", entries: 3,
            entryBytes: PromptCache.entryDiskBytes(context: 8_192) * 2, ledger: ledger, capacity: 8_192)
        backend = FamilyServingBackend(model: model, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: true)
        backend.promptCache = cache
        backend.epoch = "synthetic"
    }
    func send(_ body: [String: Any]) throws -> GenerationStatistics {
        var body = body
        body["model"] = "synthetic"
        body["temperature"] = 0
        body["reasoning_effort"] = "none"
        body["origami_return_tokens"] = true
        let request = try ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 8_192)
        return try backend.prepare(request, cancellation: GenerationControl()).run { _ in true }
    }
}

struct SyntheticFixture {
    let weights: FamilyWeights
    let tokenizer: TokenizerStack
    let folder: URL
    let opening: String, other: String
    init(_ family: ModelFamily, device: MTLDevice) throws {
        folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-bounded-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        weights = try FamilyWeights(gguf: GGUFSource(try syntheticFamilyGGUF(family, in: folder)), device: device)
        tokenizer = try tinyFamilyTokenizer("{% for m in messages %}{{ m.content }}{% endfor %}", fillers: weights.geometry.vocab - 261)
        let letters = Array("abcdefghjklmnopqrstuvwxyz0123456789,.;:-+=")
        var state: UInt64 = family == .glm ? 0xB0_0001 : 0xB0_0002
        func text(_ count: Int) -> String {
            String(
                (0..<count).map { _ in
                    state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                    return letters[Int(state >> 33) % letters.count]
                })
        }
        opening = text(900)
        other = text(700)
    }
    func server() throws -> SyntheticServer { try SyntheticServer(weights, tokenizer: tokenizer, folder: folder) }
    func remove() { try? FileManager.default.removeItem(at: folder) }
}

/// One hung GPU wait in a request, seen late (within its bound and `settle` more): `GPURetry` runs the request again,
/// and its reply, the next request's reply and the state after it are the bits of a clean run. Slow but healthy waits
/// (below the bound) change nothing. A wait never seen fails the request (`gpu_lost`), every later request in that
/// process fails at once, and a new process (a recovered device, a new backend) serves the clean bits again.
private func syntheticGPUHang(_ family: ModelFamily) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let fixture = try SyntheticFixture(family, device: device)
    defer { fixture.remove() }
    let first: [String: Any] = ["messages": [["role": "user", "content": fixture.opening]], "max_tokens": 6]
    let second: [String: Any] = ["messages": [["role": "user", "content": fixture.other]], "max_tokens": 6]
    struct Run: Equatable {
        var first: [Int] = [], second: [Int] = [], state = ""
    }
    func run(_ inject: [Double?] = []) throws -> (run: Run, retries: Int?) {
        try autoreleasepool {
            let server = try fixture.server()
            GPUWait.inject(inject)
            let one = try server.send(first)
            try require(GPUWait.injectedLeft == 0, "the request made fewer than \(inject.count) waits")
            let two = try server.send(second)
            return (Run(first: one.outputTokens ?? [], second: two.outputTokens ?? [], state: sha256(server.model)), one.gpuRetries)
        }
    }
    try withShortBounds {
        let clean = try run()
        try require(clean.run.first.count == 6 && clean.retries == nil, "the clean run: \(clean)")
        let waits = Array(repeating: Double?.some(0), count: 24)
        let late = try run(waits + [1.5])
        try require(late.retries == 1 && late.run == clean.run, "a late completion: \(late) vs \(clean)")
        let slow = try run(Array(repeating: 0.02, count: 20))
        try require(slow.retries == nil && slow.run == clean.run, "a slow healthy GPU: \(slow) vs \(clean)")
        let lost = Flag()
        GPUWait.onLost = { _ in lost.raise() }
        try autoreleasepool {
            let server = try fixture.server()
            GPUWait.inject(waits + [nil])
            let started = Date()
            do {
                _ = try server.send(first)
                throw EngineError("bounded waits: a lost GPU completed the request")
            } catch  where ServingFailure.from(error).code == "gpu_lost" {}
            try require(
                Date().timeIntervalSince(started) < 20 && lost.raised == 1, "the lost GPU took \(Date().timeIntervalSince(started)) s")
            let again = Date()
            do {
                _ = try server.send(second)
                throw EngineError("bounded waits: a request ran on a lost GPU")
            } catch  where ServingFailure.from(error).code == "gpu_lost" {}
            try require(Date().timeIntervalSince(again) < 2, "a request on the lost GPU took \(Date().timeIntervalSince(again)) s")
        }
        GPUWait.recover()
        try require(try run().run == clean.run, "the restarted model differs")
    }
}

/// A demand read that hangs (the model drive stopped answering) fails its request with `drive_stalled` at its bound;
/// while it still hangs its slot stays out of use and the next request (the same prompt) gets the clean bits; once
/// it returns, the slot is freed.
private func syntheticReadStall(_ family: ModelFamily) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let fixture = try SyntheticFixture(family, device: device)
    defer { fixture.remove() }
    let body: [String: Any] = ["messages": [["role": "user", "content": fixture.opening]], "max_tokens": 6]
    let clean = try autoreleasepool { () throws -> (reply: [Int], state: String) in
        let server = try fixture.server()
        let stats = try server.send(body)
        return (stats.outputTokens ?? [], sha256(server.model))
    }
    final class Gate: @unchecked Sendable {
        let lock = NSLock()
        var armed = true
        let open = DispatchSemaphore(value: 0)
        func probe() {
            let hold = lock.withLock {
                defer { armed = false }
                return armed
            }
            if hold { open.wait() }
        }
    }
    try withShortBounds(read: 0.5) {
        try autoreleasepool {
            let server = try fixture.server()
            let gate = Gate()
            server.runner.experts.readProbe = { _, replica in if !replica { gate.probe() } }
            // Whatever fails below, the hung read returns (an extra signal is harmless).
            defer { gate.open.signal() }
            let started = Date()
            do {
                _ = try server.send(body)
                throw EngineError("bounded waits: a hung read did not fail its request")
            } catch  where ServingFailure.from(error).code == "drive_stalled" {}
            try require(Date().timeIntervalSince(started) < 5, "the stall took \(Date().timeIntervalSince(started)) s")
            try require(server.runner.experts.stalledReads == 1, "the stalled read's slot was not held")
            let during = try server.send(body)
            try require(
                during.outputTokens ?? [] == clean.reply && sha256(server.model) == clean.state && server.runner.experts.stalledReads == 1,
                "the request during the stall differs, or freed the stuck slot")
            gate.open.signal()
            let until = Date().addingTimeInterval(5)
            while server.runner.experts.stalledReads > 0 && Date() < until {
                usleep(10_000)
                server.runner.experts.freeStalledBatches()
            }
            try require(server.runner.experts.stalledReads == 0, "the stuck slot was not freed once its read returned")
            let after = try server.send(body)
            try require(
                after.outputTokens ?? [] == clean.reply && sha256(server.model) == clean.state, "the request after the stall differs")
        }
    }
}

/// Holds background reads (`FamilyExperts.backgroundProbe`) until released, noting which records' reads began.
private final class Valve: @unchecked Sendable {
    let condition = NSCondition()
    var open = false
    var entered = Set<Int>()
    func probe(_ key: Int) {
        condition.lock()
        entered.insert(key)
        condition.broadcast()
        while !open { condition.wait() }
        condition.unlock()
    }
    /// Waits until reads of `count` records have begun.
    func began(_ count: Int) -> Bool {
        condition.lock()
        defer { condition.unlock() }
        let until = Date().addingTimeInterval(5)
        while entered.count < count && Date() < until { _ = condition.wait(until: until) }
        return entered.count >= count
    }
    func release() {
        condition.lock()
        open = true
        condition.broadcast()
        condition.unlock()
    }
}

/// Guessed reads that hang (the model drive stopped answering) bound the two waits for them: a pool resize
/// (`applyLimits`) gives up within the read bound and keeps the pool's size, and a later one does not wait again; a
/// demand fetch that needs the slots such reads hold fails with `drive_stalled` within it. Once the reads return,
/// the resize applies and the fetch succeeds.
private func syntheticGuessStall() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let fixture = try SyntheticFixture(.glm, device: device)
    defer { fixture.remove() }
    try withShortBounds(read: 0.5) {
        try autoreleasepool {
            let server = try fixture.server()
            let experts = server.runner.experts!
            let geometry = fixture.weights.geometry
            let layer = geometry.moeLayers[0]
            experts.adaptiveGuesses = false
            let full = experts.budget
            experts.setLimit(bytes: 0)  // each pool at its floor: two waves of slots
            let floor = experts.slotCount
            // The layer's pool (one per expert type) at its floor; two guesses begin (prefetches run a few reads at a
            // time) and hang, so a fetch of all but one slot's worth of other experts must wait for them.
            let pool = 2 * geometry.expertsPerToken
            let others = Array(2..<(1 + pool))
            let valve = Valve()
            experts.backgroundProbe = { valve.probe($0) }
            // Whatever fails below, the hung reads return.
            defer { valve.release() }
            let guessed = [0, 1]
            experts.prefetch(layer: layer, experts: guessed)
            try require(valve.began(guessed.count), "the guesses' reads did not begin")
            var started = Date()
            experts.setLimit(bytes: full)
            var seconds = Date().timeIntervalSince(started)
            try require(seconds >= 0.5 && seconds < 3, "a resize waited \(seconds) s for a hung guess")
            try require(experts.slotCount == floor, "a resize applied over a hung guess")
            started = Date()
            experts.setLimit(bytes: full &+ 1)
            seconds = Date().timeIntervalSince(started)
            try require(seconds < 0.3 && experts.slotCount == floor, "a second resize waited \(seconds) s for the hung guess")
            started = Date()
            do {
                _ = try experts.fetch(layer: layer, experts: others)
                throw EngineError("bounded waits: a fetch with every slot held by hung guesses succeeded")
            } catch let error as EngineError where error.code == "drive_stalled" {}
            seconds = Date().timeIntervalSince(started)
            try require(seconds < 3, "a fetch waited \(seconds) s for hung guesses")
            valve.release()
            // A guess that outlived its bound is not waited for again: the resize applies once its read has ended.
            let until = Date().addingTimeInterval(5)
            repeat {
                experts.setLimit(bytes: full)
                if experts.slotCount > floor { break }
                usleep(10_000)
            } while Date() < until
            try require(experts.slotCount > floor, "the resize did not apply once the guesses landed")
            _ = try experts.fetch(layer: layer, experts: others)
        }
    }
}

/// A request that arrives while background work holds the model on a read that hangs (the start's warm-up, a record the
/// hot copy's keeper copies under the backend's lock) fails with `drive_stalled` within the bound of a batch of reads;
/// once the read returns, the next request is served with the bits of a clean run.
private func syntheticBackgroundReadStall() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let fixture = try SyntheticFixture(.glm, device: device)
    defer { fixture.remove() }
    let body: [String: Any] = ["messages": [["role": "user", "content": fixture.opening]], "max_tokens": 6]
    let clean = try autoreleasepool { try fixture.server().send(body).outputTokens ?? [] }
    func stalls(_ server: SyntheticServer) throws {
        let started = Date()
        do {
            _ = try server.send(body)
            throw EngineError("bounded waits: a request waited for a hung background read")
        } catch  where ServingFailure.from(error).code == "drive_stalled" {}
        let seconds = Date().timeIntervalSince(started)
        try require(seconds >= 0.5 && seconds < 3, "a request waited \(seconds) s for a hung background read")
    }
    try withShortBounds(read: 0.5) {
        try autoreleasepool {
            let server = try fixture.server()
            let experts = server.runner.experts!
            let geometry = fixture.weights.geometry
            let valve = Valve()
            defer { valve.release() }
            experts.backgroundProbe = { valve.probe($0) }
            // The warm-up fills slots past two waves of a pool, which the tiny model's pools have only at an odd size.
            let full = experts.budget
            for percent in stride(from: 100, through: 20, by: -1) where experts.slotCount % 2 == 0 {
                experts.setLimit(bytes: full / 100 * UInt64(percent))
            }
            try require(experts.slotCount % 2 == 1, "no pool size leaves the warm-up room")
            let ended = DispatchSemaphore(value: 0)
            let keys = geometry.moeLayers.flatMap { l in (0..<geometry.experts).map { l * geometry.experts + $0 } }
            server.backend.warm(experts, keys: keys) { _, _ in ended.signal() }
            try require(valve.began(1), "the warm-up's read did not begin")
            try stalls(server)
            valve.release()
            try require(ended.wait(timeout: .now() + 5) == .success, "the warm-up did not end once its read returned")
            try require(try server.send(body).outputTokens ?? [] == clean, "the request after the warm-up's stall differs")
        }
        try autoreleasepool {
            let server = try fixture.server()
            let valve = Valve()
            defer { valve.release() }
            let held = DispatchSemaphore(value: 0)
            let backend = server.backend
            Thread.detachNewThread {
                // As the keeper copies a record: under the backend's lock, its read hung.
                _ = backend.withIdleHotCopy {
                    held.signal()
                    valve.probe(0)
                }
                held.signal()
            }
            try require(held.wait(timeout: .now() + 5) == .success, "the keeper's turn did not begin")
            try stalls(server)
            valve.release()
            try require(held.wait(timeout: .now() + 5) == .success, "the keeper's turn did not end once its read returned")
            try require(try server.send(body).outputTokens ?? [] == clean, "the request after the keeper's stall differs")
        }
    }
}

/// `origami_time_limit`: a reply ends at a token boundary past it with the tokens so far and finish reason `length`, the
/// bits of the same reply cut by `max_tokens`, and the follow-up continues it bitwise as it continues that one. A limit
/// that passes while the prompt is read answers 408 and keeps the conversation: the resent turn continues it.
private func syntheticTimeLimit(_ family: ModelFamily) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let fixture = try SyntheticFixture(family, device: device)
    defer { fixture.remove() }
    let forced = Array("XYZWVUTSRQXYZWVUTSRQXYZWVUTSRQXYZWVUTSRQ".utf8).map(Int.init)
    func opening(_ limit: Double?, tokens: Int = forced.count) -> [String: Any] {
        var body: [String: Any] = [
            "messages": [["role": "user", "content": fixture.opening]], "max_tokens": tokens, "origami_teacher_tokens": forced,
        ]
        if let limit { body["origami_time_limit"] = limit }
        return body
    }
    func followUp(_ reply: Int, limit: Double? = nil) -> [String: Any] {
        var body: [String: Any] = [
            "messages": [
                ["role": "user", "content": fixture.opening],
                ["role": "assistant", "content": String(decoding: forced.prefix(reply).map(UInt8.init), as: UTF8.self)],
                ["role": "user", "content": fixture.other],
            ],
            "max_tokens": 6,
        ]
        if let limit { body["origami_time_limit"] = limit }
        return body
    }
    struct Turns: Equatable {
        var reply: [Int] = [], next: [Int] = [], reused = 0, state = ""
    }
    // The opening reply's time: its prompt, and its 40 tokens.
    let (prompt, whole) = try autoreleasepool { () throws -> (Double, Double) in
        let server = try fixture.server()
        var started = Date()
        _ = try server.send(opening(nil, tokens: 1))
        let prompt = Date().timeIntervalSince(started)
        let other = try fixture.server()
        started = Date()
        _ = try other.send(opening(nil))
        return (prompt, Date().timeIntervalSince(started))
    }
    var cut: (stats: GenerationStatistics, turns: Turns)?
    for share in [0.5, 0.3, 0.7] where cut == nil {
        try autoreleasepool {
            let server = try fixture.server()
            let stats = try server.send(opening(prompt + share * (whole - prompt)))
            guard (1..<forced.count).contains(stats.completionTokens) else { return }
            let next = try server.send(followUp(stats.completionTokens))
            cut = (
                stats,
                Turns(
                    reply: stats.outputTokens ?? [], next: next.outputTokens ?? [], reused: next.cachedTokens, state: sha256(server.model))
            )
        }
    }
    guard let cut else { throw EngineError("bounded waits: the time limit never ended the reply between its first and last token") }
    try require(cut.stats.finishReason == "length", "the finish reason was \(cut.stats.finishReason)")
    let k = cut.stats.completionTokens
    let reference = try autoreleasepool { () throws -> Turns in
        let server = try fixture.server()
        let stats = try server.send(opening(nil, tokens: k))
        let next = try server.send(followUp(k))
        return Turns(reply: stats.outputTokens ?? [], next: next.outputTokens ?? [], reused: next.cachedTokens, state: sha256(server.model))
    }
    try require(cut.turns == reference && reference.reused > 0, "cut at \(k) tokens: \(cut.turns) vs max_tokens \(reference)")
    // A limit already past when the follow-up's prompt is read: 408, and the resent follow-up continues the conversation.
    let resent = try autoreleasepool { () throws -> Turns in
        let server = try fixture.server()
        _ = try server.send(opening(nil, tokens: k))
        do {
            _ = try server.send(followUp(k, limit: 0))
            throw EngineError("bounded waits: a past limit read the prompt")
        } catch let failure as ServingFailure where failure.status == 408 && failure.code == "request_timeout" {}
        let next = try server.send(followUp(k))
        return Turns(reply: reference.reply, next: next.outputTokens ?? [], reused: next.cachedTokens, state: sha256(server.model))
    }
    try require(resent == reference, "the follow-up after a timed-out prompt: \(resent) vs \(reference)")
}

/// Qwen's embedding and PLE-row prefetches whose drive stopped answering (one row read hangs): each wait fails with
/// `drive_stalled` at the demand read bound, the PLE wait lets its token's prefetch hold go, and the next prefetch on the
/// same ledger succeeds while the hung reads are still out.
private func qwenRowPrefetchStallBounded() throws {
    final class Gate: @unchecked Sendable {
        let opened = DispatchSemaphore(value: 0)
        let lock = NSLock()
        var held = 0
        func hang() {
            lock.withLock { held += 1 }
            opened.wait()
        }
    }
    let ledger = try MemoryLedger(ceiling: 1_000_000_000, transientMargin: 1_000_000)
    let layout = EmbeddingRowPrefetch.Layout(wordBytes: 1280, scaleBytes: 80, biasBytes: 80)
    let limits = IOScheduler.Limits(
        maximumConcurrent: 1, reservedDemandConcurrent: 1, maximumOutstanding: 4,
        reservedDemandOutstanding: 4, maximumOutstandingBytes: 32_000_000, reservedDemandBytes: 32_000_000)
    let scheduler = try IOScheduler(ledger: ledger, internalLimits: limits, externalLimits: limits, readHygiene: true)
    defer { scheduler.shutdown() }
    let gate = Gate()
    defer { for _ in 0..<2 { gate.opened.signal() } }
    try withShortBounds(read: 0.3) {
        func stalls(_ what: String, _ wait: () throws -> Double) throws {
            let started = Date()
            // A Result, not do/catch: the debug build's module pass warns that this catch is unreachable.
            switch Result(catching: wait) {
            case .success: throw EngineError("bounded waits: a hung \(what) read was not bounded")
            case .failure(let error): guard (error as? EngineError)?.code == "drive_stalled" else { throw error }
            }
            let seconds = Date().timeIntervalSince(started)
            try require(seconds >= 0.3 && seconds < 5, "the \(what) stall took \(seconds) s")
        }
        let fill: @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void = { id, buffer in
            for i in buffer.indices { buffer[i] = UInt8(truncatingIfNeeded: id &+ i) }
        }
        let embedding = try EmbeddingRowPrefetch(ids: Array(0..<100), layout: layout, ledger: ledger) { id, buffer in
            if id == 42 { gate.hang() }
            try fill(id, buffer)
        }
        try stalls("embedding row") { try embedding.wait(shouldContinue: { true }) }
        let next = try EmbeddingRowPrefetch(ids: Array(0..<100), layout: layout, ledger: ledger, read: fill)
        _ = try next.wait(shouldContinue: { true })
        try require(try next.embedding().count == 100 * 2560, "the embedding prefetch after a stall")

        let hold = scheduler.holdPrefetch()
        try require(scheduler.hygieneCounts()?.liveHolds == 1, "the fixture's prefetch hold")
        let rows = (0..<8).map { token in (0..<PLERowPrefetch.rowsPerToken).map { token * 16 + $0 } }
        let ple = try PLERowPrefetch(tokenRows: rows, ledger: ledger, hold: hold) { index, buffer in
            if index == 70 { gate.hang() }
            try fill(index, buffer)
        }
        try stalls("PLE row") { try ple.wait(shouldContinue: { true }) }
        try require(scheduler.hygieneCounts()?.liveHolds == 0, "the stalled PLE wait kept its token's prefetch hold")
        let again = try PLERowPrefetch(tokenRows: rows, ledger: ledger, read: fill)
        _ = try again.wait(shouldContinue: { true })
        try require(try again.packed(tokens: 0..<8).count == 8 * 16 * PLERowPrefetch.rowBytes, "the PLE prefetch after a stall")
        try require(gate.lock.withLock { gate.held } == 2, "both hung reads are still out")
    }
}
