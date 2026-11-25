import Foundation

/// One model at a time for `origami serve`'s front (the CLI's `ModelFront`): requests name the model they want, and a
/// request for another model than the running one waits until the running requests finish, then the scheduler stops
/// that model and starts the named one. Requests are taken in arrival order: once a request waits for another model,
/// later requests for the running one wait behind it, so a steady stream of one model's requests never starves
/// another's. A request that names no model (or one not listed) uses whichever model runs, else `fallback`.
/// With `batch` (the front's: alternating models), a request for the loaded model may go ahead of an earlier one for
/// another, so the requests that wait for the loaded model are served before a switch costs a reload and a cold cache:
/// at most `batch` such admissions per switch, and none once the other has waited `batchSeconds`, so it is not starved.
/// `start` and `stop` run outside the lock, one switch at a time. A switch asks `ready` first, while the old model still
/// serves, so a target that cannot start (its drive gone…) never costs the working one; and a target that fails to start
/// after the old one stopped brings the old one back.
public final class ModelScheduler<Child: AnyObject>: @unchecked Sendable {
    public struct Gone: Error { public init() {} }
    private let condition = NSCondition()
    private let start: (String) throws -> Child
    private let ready: (String) throws -> Void
    private let stop: (Child) -> Void
    private let alive: (Child) -> Bool
    private let batch: Int, batchSeconds: TimeInterval
    private var running: (id: String, child: Child)?
    private var inFlight = 0, switching = false, closed = false, nextTicket = 0
    private var pendingReloads = 0
    /// Admissions for the loaded model that went ahead of a request for another since the last switch (`batch`).
    private var wentAhead = 0
    /// Health, status and dashboard reads lent the running model (`acquireRunning`): never part of a drain, so they
    /// keep answering while a reload waits for the running reply; a stop waits only for those already lent.
    private var lent = 0
    private var waiting: [(ticket: Int, want: String?, since: Date)] = []
    /// Models `holding` keeps from starting (nil: every model).
    private var holds: [(ticket: Int, ids: Set<String>?)] = []

    public init(
        start: @escaping (String) throws -> Child, ready: @escaping (String) throws -> Void = { _ in }, stop: @escaping (Child) -> Void,
        alive: @escaping (Child) -> Bool, batch: Int = 0, batchSeconds: TimeInterval = 0
    ) {
        self.start = start
        self.ready = ready
        self.stop = stop
        self.alive = alive
        self.batch = batch
        self.batchSeconds = batchSeconds
    }

    /// The running model's id and child, or nil (none yet, or between two models).
    public var current: (id: String, child: Child)? {
        condition.lock()
        defer { condition.unlock() }
        return running
    }
    /// Requests being served (not status reads): a reload waits for them.
    public var busy: Bool {
        condition.lock()
        defer { condition.unlock() }
        return inFlight > 0
    }
    /// New admissions are closed while reload/unload drains its already admitted work.
    public var isDraining: Bool {
        condition.lock()
        defer { condition.unlock() }
        return pendingReloads > 0
    }

    /// Waits until `want` (nil: whichever model runs, else `fallback`) serves, starting it when needed, and counts the
    /// caller in until `release()`. `gone` is asked about once a second while waiting: true gives up with `Gone`, as
    /// does a scheduler stopped by `stopAll`. A failed start throws to this caller; the next request for that model
    /// tries again.
    public func acquire(_ want: String?, fallback: String, gone: () -> Bool = { false }) throws -> (id: String, child: Child) {
        condition.lock()
        defer { condition.unlock() }
        let ticket = nextTicket
        nextTicket += 1
        waiting.append((ticket, want, Date()))
        defer {
            waiting.removeAll { $0.ticket == ticket }
            condition.broadcast()
        }
        while true {
            guard !closed else { throw Gone() }
            let target = want ?? running?.id ?? fallback
            let other = waiting.prefix { $0.ticket != ticket }.first { ($0.want ?? target) != target }
            // The loaded model's request may go ahead of the other model's while the batch bounds allow.
            let ahead =
                other != nil && running?.id == target && wentAhead < batch
                && Date().timeIntervalSince(other!.since) < batchSeconds
            if !switching && pendingReloads == 0 && (other == nil || ahead) {
                if let running, running.id == target, alive(running.child) {
                    inFlight += 1
                    wentAhead = other == nil ? 0 : wentAhead + 1
                    return running
                }
                if other == nil && inFlight == 0 && !holds.contains(where: { $0.ids?.contains(target) ?? true }) {
                    switching = true
                    wentAhead = 0
                    let old = running
                    if let old, old.id != target {
                        condition.unlock()
                        let checked = Result { try ready(target) }
                        condition.lock()
                        if case .failure(let error) = checked {
                            switching = false
                            throw error
                        }
                    }
                    running = nil
                    while lent > 0 { condition.wait() }
                    condition.unlock()
                    if let old { stop(old.child) }
                    let started = Result { try start(target) }
                    // The old model comes back after a failed start; the start error still goes to this request.
                    let restored = old.flatMap { old in (try? started.get()) == nil && old.id != target ? try? start(old.id) : nil }
                    condition.lock()
                    switching = false
                    condition.broadcast()
                    switch started {
                    case .success(let child): running = (target, child)
                    case .failure(let error):
                        running = old.flatMap { old in restored.map { (old.id, $0) } }
                        throw error
                    }
                    continue
                }
            }
            if gone() { throw Gone() }
            _ = condition.wait(until: Date().addingTimeInterval(1))
        }
    }

    /// The running model for a request that needs no particular one (health, status, the dashboard), lent until
    /// `releaseRunning()`; nil while none runs or one is being stopped. It does not queue behind requests waiting for
    /// another model, and a reload waiting for the running reply still lends the model that serves until it stops.
    public func acquireRunning() -> (id: String, child: Child)? {
        condition.lock()
        defer { condition.unlock() }
        guard !closed, let running, alive(running.child) else { return nil }  // a switch clears `running` once it stops it
        lent += 1
        return running
    }

    /// Ends a request counted in by `acquire`.
    public func release() {
        condition.lock()
        inFlight -= 1
        condition.broadcast()
        condition.unlock()
    }
    /// Ends a read lent by `acquireRunning`.
    public func releaseRunning() {
        condition.lock()
        lent -= 1
        condition.broadcast()
        condition.unlock()
    }

    /// Restarts with `id` (the same model or another) once the running requests finish, as a switch does, so the model
    /// re-reads its settings; requests that arrive meanwhile wait for it. `between` runs while no model does (a move of
    /// the files the running one reads); with no `id` none starts after it, and the next request starts its own.
    public func reload(_ id: String?, between: () -> Void = {}) throws {
        condition.lock()
        defer { condition.unlock() }
        pendingReloads += 1
        defer {
            pendingReloads -= 1
            condition.broadcast()
        }
        while switching || inFlight > 0 {
            guard !closed else { throw Gone() }
            condition.wait()
        }
        guard !closed else { throw Gone() }
        switching = true
        let old = running
        running = nil
        while lent > 0 { condition.wait() }
        condition.unlock()
        if let old { stop(old.child) }
        between()
        let started = id.map { id in Result { (id, try start(id)) } }
        condition.lock()
        switching = false
        condition.broadcast()
        running = try started?.get()
    }

    /// Stop and unload: waits up to `seconds` for the running requests (and a start under way), then calls `cancel` once
    /// if some still run (the front closes their connections, which cancels them), and stops the model once they end.
    /// None starts after it: the next request starts its own, as after `reload(nil)`. Returns how many it cancelled.
    public func unload(within seconds: TimeInterval, cancel: () -> Void) throws -> Int {
        condition.lock()
        pendingReloads += 1
        defer {
            condition.lock()
            pendingReloads -= 1
            condition.broadcast()
            condition.unlock()
        }
        let deadline = Date().addingTimeInterval(seconds)
        while (switching || inFlight > 0) && !closed && condition.wait(until: deadline) {}
        let busy = inFlight
        condition.unlock()
        if busy > 0 { cancel() }
        try reload(nil)
        return busy
    }

    /// Runs `body` while none of `ids` (nil: no model) starts: a request for one waits until it ends (a move or removal
    /// of its files), while the running model keeps serving its own. A start under way finishes first, so `body` never
    /// meets a model that is opening its files while `current` is still nil.
    public func holding<T>(_ ids: Set<String>?, _ body: () throws -> T) rethrows -> T {
        condition.lock()
        while switching && !closed { condition.wait() }
        let ticket = nextTicket
        nextTicket += 1
        holds.append((ticket, ids))
        condition.unlock()
        defer {
            condition.lock()
            holds.removeAll { $0.ticket == ticket }
            condition.broadcast()
            condition.unlock()
        }
        return try body()
    }

    /// Shutdown: stops the running model; waiting and later requests give up with `Gone`.
    public func stopAll() {
        condition.lock()
        while switching { condition.wait() }
        let old = running
        running = nil
        closed = true
        condition.broadcast()
        condition.unlock()
        if let old { stop(old.child) }
    }

    /// The listed id a requested model name means: the id itself, the id with a `-<level>` suffix (`-fast`,
    /// `-faster`…), either with Ollama's `:latest` tag; nil for a name not listed.
    public static func resolve(_ name: String?, ids: [String]) -> String? {
        guard var name else { return nil }
        if name.hasSuffix(":latest") { name.removeLast(":latest".count) }
        if ids.contains(name) { return name }
        let suffixes = QualityDial.levels.dropFirst().map(\.name) + QualityDial.aliases.keys
        for suffix in suffixes where name.hasSuffix("-" + suffix) {
            let base = String(name.dropLast(suffix.count + 1))
            if ids.contains(base) { return base }
        }
        return nil
    }
}
