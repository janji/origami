import Foundation
import OrigamiEngine

// `origami serve`'s front (`ModelScheduler`): one model at a time; a switch waits for the running requests; requests
// are served in arrival order; a request for no model uses whichever runs; a failed start throws and the next request
// tries again, and the model it replaced comes back; a target that is not ready never stops the working one; a reload waits for running requests and restarts (to none: starts none until asked), while status reads
// still get the model that serves until it stops, without extending the wait; an unload waits
// its time for them, then cancels them and stops the model; a hold keeps a
// model from starting; shutdown refuses
// the rest; requested names resolve with level suffixes and Ollama's `:latest`.
func modelSchedulerChecks() -> [(String, () throws -> Void)] {
    [
        ("model_scheduler_one_at_a_time", modelSchedulerOneAtATime),
        ("model_scheduler_resolves_names", modelSchedulerResolvesNames),
        ("model_scheduler_reload_closes_admission", modelSchedulerReloadClosesAdmission),
        ("model_scheduler_keeps_working_model", modelSchedulerKeepsWorkingModel),
        ("model_scheduler_hold_and_start_never_overlap", modelSchedulerHoldAndStartNeverOverlap),
        ("model_scheduler_batches_loaded_model", modelSchedulerBatchesLoadedModel),
    ]
}

private func require(_ condition: Bool, _ message: @autoclosure () -> String) throws {
    if !condition { throw EngineError("Model scheduler: " + message()) }
}

private final class FakeChild {
    let id: String
    var stopped = false
    init(_ id: String) { self.id = id }
}
private final class Journal: @unchecked Sendable {
    let lock = NSLock()
    var events: [String] = [], running = 0, most = 0, failNext = false
    func add(_ event: String) { lock.withLock { events.append(event) } }
    var snapshot: [String] { lock.withLock { events } }
}

private func modelSchedulerOneAtATime() throws {
    let journal = Journal()
    let scheduler = ModelScheduler<FakeChild>(
        start: { id in
            try journal.lock.withLock {
                if journal.failNext {
                    journal.failNext = false
                    throw EngineError("no memory for \(id)")
                }
                journal.events.append("start " + id)
                journal.running += 1
                journal.most = max(journal.most, journal.running)
            }
            return FakeChild(id)
        },
        stop: { child in
            journal.lock.withLock {
                journal.events.append("stop " + child.id)
                journal.running -= 1
                child.stopped = true
            }
        }, alive: { !$0.stopped })
    func later(_ want: String) -> DispatchSemaphore {
        let done = DispatchSemaphore(value: 0)
        Thread.detachNewThread {
            if let got = try? scheduler.acquire(want, fallback: "a") {
                journal.add("served " + got.id)
                scheduler.release()
            }
            done.signal()
        }
        return done
    }

    // No model named: the fallback starts, and stays for the next such request.
    try require(try scheduler.acquire(nil, fallback: "a").id == "a", "the fallback did not start")
    scheduler.release()
    try require(try scheduler.acquire(nil, fallback: "z").id == "a" && journal.snapshot == ["start a"], "a request for no model switched")
    scheduler.release()
    // b waits while a request for a runs; a later request for a waits behind b, so a steady stream of a cannot starve b.
    _ = try scheduler.acquire("a", fallback: "a")
    let b = later("b")
    usleep(100_000)
    let a = later("a")
    usleep(100_000)
    try require(journal.snapshot == ["start a"], "a switch did not wait for the running request: \(journal.snapshot)")
    scheduler.release()
    b.wait()
    a.wait()
    try require(
        journal.snapshot == ["start a", "stop a", "start b", "served b", "stop b", "start a", "served a"] && journal.most == 1,
        "switches out of order or overlapping: \(journal.snapshot), most \(journal.most)")
    // A failed start throws to its request and brings the replaced model back; the next one tries again.
    journal.lock.withLock { journal.failNext = true }
    try require(
        (try? scheduler.acquire("c", fallback: "a")) == nil && scheduler.current?.id == "a"
            && journal.snapshot.suffix(2) == ["stop a", "start a"], "a failed start served or lost the old model: \(journal.snapshot)")
    try require(try scheduler.acquire("c", fallback: "a").id == "c", "a start was not retried")
    scheduler.release()
    // Health and status take the running model without queueing; a client that left gives up.
    try require(scheduler.acquireRunning()?.id == "c", "the running model was not lent")
    scheduler.releaseRunning()
    _ = try scheduler.acquire("c", fallback: "a")
    do {
        _ = try scheduler.acquire("d", fallback: "a") { true }
        throw EngineError("Model scheduler: a client that left was served")
    } catch is ModelScheduler<FakeChild>.Gone {}
    scheduler.release()
    // A reload waits for the running request, then restarts the model (settings read again) with one up at a time.
    let held = try scheduler.acquire("c", fallback: "a")
    let reloaded = DispatchSemaphore(value: 0)
    Thread.detachNewThread {
        if (try? scheduler.reload("c")) != nil { journal.add("reloaded") }
        reloaded.signal()
    }
    usleep(100_000)
    try require(!journal.snapshot.contains("reloaded"), "a reload did not wait for the running request")
    _ = held
    scheduler.release()
    reloaded.wait()
    try require(
        journal.snapshot.suffix(3) == ["stop c", "start c", "reloaded"] && journal.most == 1 && scheduler.current?.id == "c",
        "reload order: \(journal.snapshot)")
    // A reload to no model (a move of what no model then reads) starts none while it runs: a request waits, then starts its own.
    var waiter: DispatchSemaphore?
    try scheduler.reload(nil) {
        waiter = later("c")
        usleep(100_000)
        journal.add("between")
    }
    waiter?.wait()
    try require(
        journal.snapshot.suffix(4) == ["stop c", "between", "start c", "served c"] && journal.most == 1,
        "reload to none: \(journal.snapshot)")
    // A hold (a move or removal of a model's files) keeps that model from starting; the running one keeps serving.
    var waiting: DispatchSemaphore?
    try scheduler.holding(["d"]) {
        try require(try scheduler.acquire("c", fallback: "a").id == "c", "a hold stopped the running model")
        scheduler.release()
        waiting = later("d")
        usleep(100_000)
        try require(!journal.snapshot.contains("start d"), "a held model started: \(journal.snapshot)")
    }
    waiting?.wait()
    try require(
        journal.snapshot.suffix(3) == ["stop c", "start d", "served d"] && journal.most == 1, "after the hold: \(journal.snapshot)")
    // Stop and unload: with nothing running it stops the model at once; a running request gets its wait, then is
    // cancelled (here its holder releases when told), and the model stops only after it ends.
    try require(
        try scheduler.unload(within: 5) { journal.add("cancel") } == 0 && journal.snapshot.suffix(1) == ["stop d"]
            && scheduler.current == nil,
        "an idle unload: \(journal.snapshot)")
    _ = try scheduler.acquire("d", fallback: "a")
    let started = Date()
    let cancelled = try scheduler.unload(within: 0.3) {
        journal.add("cancel")
        scheduler.release()
    }
    try require(
        cancelled == 1 && Date().timeIntervalSince(started) >= 0.3 && journal.snapshot.suffix(3) == ["start d", "cancel", "stop d"]
            && scheduler.current == nil,
        "a busy unload: \(journal.snapshot), cancelled \(cancelled)")
    // A request that finishes within the wait is not cancelled.
    _ = try scheduler.acquire("d", fallback: "a")
    Thread.detachNewThread {
        usleep(100_000)
        scheduler.release()
    }
    try require(
        try scheduler.unload(within: 5) { journal.add("cancel") } == 0 && journal.snapshot.suffix(2) == ["start d", "stop d"],
        "an unload cancelled a request that finished in time: \(journal.snapshot)")
    _ = try scheduler.acquire("d", fallback: "a")
    scheduler.release()
    // Shutdown stops the running model and refuses the rest.
    scheduler.stopAll()
    try require(
        journal.snapshot.suffix(1) == ["stop d"] && journal.running == 0 && scheduler.acquireRunning() == nil,
        "shutdown left a model running: \(journal.snapshot)")
    do {
        _ = try scheduler.acquire("a", fallback: "a")
        throw EngineError("Model scheduler: served after shutdown")
    } catch is ModelScheduler<FakeChild>.Gone {}
}

private func modelSchedulerResolvesNames() throws {
    let ids = ["qwen3.8-flash-next", "glm-2bit"]
    let cases: [(String?, String?)] = [
        ("glm-2bit", "glm-2bit"), ("glm-2bit-faster", "glm-2bit"), ("glm-2bit-fast", "glm-2bit"), ("glm-2bit:latest", "glm-2bit"),
        ("qwen3.8-flash-next-dynamic:latest", "qwen3.8-flash-next"), ("qwen3.8-flash-next-fastest", "qwen3.8-flash-next"),
        ("gpt-4o", nil), ("glm-2bit-turbo", nil), (nil, nil), ("", nil),
    ]
    for (name, expected) in cases {
        try require(
            ModelScheduler<FakeChild>.resolve(name, ids: ids) == expected,
            "\(name ?? "nil") resolved to \(ModelScheduler<FakeChild>.resolve(name, ids: ids) ?? "nil")")
    }
}

private func modelSchedulerReloadClosesAdmission() throws {
    let scheduler = ModelScheduler<FakeChild>(start: { FakeChild($0) }, stop: { $0.stopped = true }, alive: { !$0.stopped })
    _ = try scheduler.acquire("a", fallback: "a")
    let reloaded = DispatchSemaphore(value: 0)
    let admitted = DispatchSemaphore(value: 0)
    Thread.detachNewThread {
        try? scheduler.reload("a")
        reloaded.signal()
    }
    let deadline = Date().addingTimeInterval(2)
    while !scheduler.isDraining && Date() < deadline { usleep(1_000) }
    try require(scheduler.isDraining, "reload never closed admission")
    // Status keeps answering from the model that still serves, without extending the drain.
    try require(scheduler.acquireRunning()?.id == "a", "a pending reload hid the serving model from status")
    Thread.detachNewThread {
        if (try? scheduler.acquire("a", fallback: "a")) != nil {
            scheduler.release()
            admitted.signal()
        }
    }
    try require(admitted.wait(timeout: .now() + 0.05) == .timedOut, "later same-model request bypassed reload")
    scheduler.release()
    // The reply ended: no new status read is lent the model being stopped, and the stop waits for the one lent.
    let deadline2 = Date().addingTimeInterval(2)
    while scheduler.acquireRunning().map({ _ in scheduler.releaseRunning() }) != nil && Date() < deadline2 { usleep(1_000) }
    try require(scheduler.acquireRunning() == nil, "a status read was lent a model being stopped")
    try require(reloaded.wait(timeout: .now() + 0.05) == .timedOut, "the model stopped under a status read")
    scheduler.releaseRunning()
    try require(reloaded.wait(timeout: .now() + 2) == .success, "reload starved after the original request ended")
    try require(admitted.wait(timeout: .now() + 2) == .success && !scheduler.isDraining, "admission did not reopen")
    scheduler.stopAll()
}

private func modelSchedulerKeepsWorkingModel() throws {
    let journal = Journal()
    let scheduler = ModelScheduler<FakeChild>(
        start: { id in
            guard id != "broken" else { throw EngineError("broken exited") }
            journal.add("start " + id)
            return FakeChild(id)
        },
        ready: { id in
            guard id != "unplugged" else { throw EngineError("drive not connected") }
        },
        stop: { child in
            journal.add("stop " + child.id)
            child.stopped = true
        }, alive: { !$0.stopped })
    _ = try scheduler.acquire("a", fallback: "a")
    scheduler.release()
    // A target that cannot start: its error, and the working model was never stopped.
    do {
        _ = try scheduler.acquire("unplugged", fallback: "a")
        throw EngineError("Model scheduler: a target that is not ready was served")
    } catch let error as EngineError where "\(error)".contains("drive not connected") {}
    try require(
        journal.snapshot == ["start a"] && scheduler.current?.id == "a", "a target not ready stopped the model: \(journal.snapshot)")
    // A target that fails once the old one stopped: its error, and the old one serves again.
    do {
        _ = try scheduler.acquire("broken", fallback: "a")
        throw EngineError("Model scheduler: a broken target was served")
    } catch let error as EngineError where "\(error)".contains("broken exited") {}
    try require(
        journal.snapshot == ["start a", "stop a", "start a"] && scheduler.current?.id == "a",
        "the old model did not come back: \(journal.snapshot)")
    try require(try scheduler.acquire(nil, fallback: "z").id == "a", "the restored model does not serve")
    scheduler.release()
    scheduler.stopAll()
}

/// A storage hold and a model start never overlap, in either order: a hold that arrives while a start is opening the
/// files waits for it to finish (and then sees the model running); a start that arrives during a hold waits for the hold.
private func modelSchedulerHoldAndStartNeverOverlap() throws {
    let journal = Journal()
    let gate = DispatchSemaphore(value: 0)
    let scheduler = ModelScheduler<FakeChild>(
        start: { id in
            journal.add("starting \(id)")
            if id == "a" { gate.wait() }
            journal.add("started \(id)")
            return FakeChild(id)
        }, stop: { _ in }, alive: { _ in true })
    // Start before hold.
    let served = DispatchSemaphore(value: 0)
    Thread.detachNewThread {
        if (try? scheduler.acquire("a", fallback: "a")) != nil { scheduler.release() }
        served.signal()
    }
    while !journal.snapshot.contains("starting a") { usleep(1_000) }
    let held = DispatchSemaphore(value: 0)
    Thread.detachNewThread {
        scheduler.holding(["a"]) { journal.add("hold sees \(scheduler.current?.id ?? "none")") }
        held.signal()
    }
    usleep(100_000)
    try require(!journal.snapshot.contains { $0.hasPrefix("hold") }, "a hold ran while a start opened the files: \(journal.snapshot)")
    gate.signal()
    held.wait()
    served.wait()
    try require(
        journal.snapshot.firstIndex(of: "started a")! < journal.snapshot.firstIndex(of: "hold sees a")!,
        "start before hold: \(journal.snapshot)")
    // Hold before start.
    let inside = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)
    Thread.detachNewThread {
        scheduler.holding(["b"]) {
            inside.signal()
            release.wait()
            journal.add("hold ends")
        }
    }
    inside.wait()
    let switched = DispatchSemaphore(value: 0)
    Thread.detachNewThread {
        if (try? scheduler.acquire("b", fallback: "a")) != nil { scheduler.release() }
        switched.signal()
    }
    usleep(100_000)
    try require(!journal.snapshot.contains("starting b"), "a start ran during a hold: \(journal.snapshot)")
    release.signal()
    switched.wait()
    try require(
        journal.snapshot.firstIndex(of: "hold ends")! < journal.snapshot.firstIndex(of: "starting b")!,
        "hold before start: \(journal.snapshot)")
}

/// With `batch`: the loaded model's requests go ahead of one waiting for another model, at most `batch` of them, and
/// none once that one has waited `batchSeconds`; then it switches.
private func modelSchedulerBatchesLoadedModel() throws {
    let journal = Journal()
    let scheduler = ModelScheduler<FakeChild>(
        start: { id in
            journal.add("start " + id)
            return FakeChild(id)
        },
        stop: { child in
            journal.add("stop " + child.id)
            child.stopped = true
        }, alive: { !$0.stopped }, batch: 2, batchSeconds: 60)
    func later(_ want: String) -> DispatchSemaphore {
        let done = DispatchSemaphore(value: 0)
        Thread.detachNewThread {
            if let got = try? scheduler.acquire(want, fallback: "a") {
                journal.add("served " + got.id)
                scheduler.release()
            }
            done.signal()
        }
        return done
    }
    _ = try scheduler.acquire("a", fallback: "a")
    let b = later("b")
    usleep(100_000)
    try require(later("a").wait(timeout: .now() + 2) == .success, "a request for the loaded model did not go ahead")
    try require(later("a").wait(timeout: .now() + 2) == .success, "a second request for the loaded model did not go ahead")
    let third = later("a")
    usleep(200_000)
    try require(journal.snapshot == ["start a", "served a", "served a"], "more than the batch went ahead: \(journal.snapshot)")
    scheduler.release()
    b.wait()
    third.wait()
    try require(
        journal.snapshot == ["start a", "served a", "served a", "stop a", "start b", "served b", "stop b", "start a", "served a"],
        "batch order: \(journal.snapshot)")
    // Once the other has waited `batchSeconds` (here 0 s), none goes ahead: the strict order.
    let strict = ModelScheduler<FakeChild>(
        start: { FakeChild($0) }, stop: { $0.stopped = true }, alive: { !$0.stopped }, batch: 2, batchSeconds: 0)
    _ = try strict.acquire("a", fallback: "a")
    let other = DispatchSemaphore(value: 0)
    Thread.detachNewThread {
        if (try? strict.acquire("b", fallback: "a")) != nil { strict.release() }
        other.signal()
    }
    usleep(100_000)
    let same = DispatchSemaphore(value: 0)
    Thread.detachNewThread {
        if (try? strict.acquire("a", fallback: "a")) != nil { strict.release() }
        same.signal()
    }
    try require(same.wait(timeout: .now() + 0.3) == .timedOut, "a request went ahead past batchSeconds")
    strict.release()
    other.wait()
    same.wait()
    scheduler.stopAll()
    strict.stopAll()
}
