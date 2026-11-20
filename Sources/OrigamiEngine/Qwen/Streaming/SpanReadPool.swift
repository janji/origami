import Foundation

/// Dedicated reader threads for the nine original spans of one expert record (gate/up/down x
/// weight/scales/biases). Read one after another, a record costs about 2.3 ms on the internal SSD;
/// issued together, about 1.2 ms (nine-span bench on the development Mac). GCD's pool does not provide enough
/// concurrent blocking preads (the PLE row lesson), so the pool owns its threads. Each task writes a
/// disjoint destination range; `run` returns only after every task has stopped, and throws the first
/// error. Bytes are unchanged: the same spans, the same validated reads.
///
/// The workers share only `State` with the pool, never the pool itself, so releasing the pool (or
/// `stop()`) wakes every idle worker, which then exits; a pool no longer leaks its threads for the
/// life of the process. Tasks already queued still run before a worker exits.
///
/// Priority (`decode_prerouting_read_hygiene`): a `run` called from a predicted (prefetch) read
/// carries that read's IOReadPriority; workers take the first queued item that is not deferrable
/// (demand spans, PLE rows, promoted predictions) and only then the oldest deferrable one. With
/// no tokens (hygiene off) the first item is always taken: plain FIFO, exactly as before.
public final class SpanReadPool: @unchecked Sendable {
    private final class State: @unchecked Sendable {
        let condition = NSCondition()
        var queue: [(task: () throws -> Void, done: (Error?) -> Void, priority: IOReadPriority?)] = []
        var stopped = false
        func stop() {
            condition.lock()
            stopped = true
            condition.broadcast()
            condition.unlock()
        }
        /// Condition locked, queue non-empty. FIFO among urgent items, then FIFO among deferrable ones.
        func takeLocked() -> (task: () throws -> Void, done: (Error?) -> Void, priority: IOReadPriority?) {
            guard queue[0].priority?.isDeferrable == true,
                let index = queue.firstIndex(where: { $0.priority?.isDeferrable != true })
            else { return queue.removeFirst() }
            queue[0].priority?.recordPoolDeferral()
            return queue.remove(at: index)
        }
    }
    private let state = State()
    let threads: Int

    public init(threads: Int = 16) throws {
        guard (1...64).contains(threads) else { throw EngineError("Span read pool size must be 1...64") }
        self.threads = threads
        for _ in 0..<threads {
            let state = self.state
            let thread = Thread {
                while true {
                    state.condition.lock()
                    while state.queue.isEmpty && !state.stopped { state.condition.wait() }
                    if state.queue.isEmpty {
                        state.condition.unlock()
                        return
                    }
                    let item = state.takeLocked()
                    state.condition.unlock()
                    do {
                        try item.task()
                        item.done(nil)
                    } catch { item.done(error) }
                }
            }
            thread.stackSize = 512 * 1024
            thread.qualityOfService = .userInitiated
            thread.start()
        }
    }

    deinit { state.stop() }

    /// Wake and retire every idle worker; queued tasks still complete. Later `run` calls are refused
    /// and later `enqueue` calls execute inline on the caller.
    package func stop() { state.stop() }

    package var isStopped: Bool {
        state.condition.lock()
        defer { state.condition.unlock() }
        return state.stopped
    }

    /// Items queued and not yet taken by a worker (standalone checks).
    package var queuedTasks: Int {
        state.condition.lock()
        defer { state.condition.unlock() }
        return state.queue.count
    }

    /// Queue one task without waiting; the caller owns its own join (for example a DispatchGroup).
    package func enqueue(_ work: @escaping () -> Void) {
        state.condition.lock()
        if state.stopped {
            state.condition.unlock()
            work()
            return
        }
        state.queue.append(({ work() }, { _ in }, nil))
        state.condition.signal()
        state.condition.unlock()
    }

    /// Run all tasks concurrently and join them; the first error is rethrown after every task stopped.
    package func run(_ tasks: [() throws -> Void]) throws {
        try run(tasks, priority: IOReadPriority.current)
    }

    /// `run` with an explicit read priority (nil = urgent); see the type comment.
    package func run(_ tasks: [() throws -> Void], priority: IOReadPriority?) throws {
        let group = DispatchGroup()
        let errors = NSLock()
        var first: Error?
        state.condition.lock()
        guard !state.stopped else {
            state.condition.unlock()
            throw EngineError("Span read pool is stopped")
        }
        for task in tasks {
            group.enter()
            state.queue.append(
                (
                    task,
                    { error in
                        if let error {
                            errors.lock()
                            if first == nil { first = error }
                            errors.unlock()
                        }
                        group.leave()
                    }, priority
                ))
        }
        state.condition.broadcast()
        state.condition.unlock()
        group.wait()
        if let first { throw first }
    }
}
