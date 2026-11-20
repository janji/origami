import Foundation
import Metal

/// Bounded waits on command buffers (ROADMAP R2): a GPU whose completion never arrives fails the request instead of
/// hanging the server. `waitBounded` is `waitUntilCompleted` with a bound:
///
///     bound = max(floor, multiple × work × CommandBudget.seconds / budget.limit)
///
/// `budget.limit` is the work this device runs in `CommandBudget.seconds` (fitted to measured GPU time, halved after an
/// abort, never above its start), so `work × seconds / limit` is no less than the expected GPU time, on any Mac: no
/// machine constant is needed. Work the caller does not count is bounded by `floor` alone (decode steps of milliseconds).
/// A buffer still running at its bound is waited for `settle` times longer, so nothing reuses the buffers it writes:
/// - it ends in that time: `gpu_stalled`, which `GPURetry` runs again from saved state like macOS's own GPU timeout
///   (the budget halves);
/// - it does not: the device is taken for lost (`gpu_lost`, by the watchdog while the wait still blocks): every later
///   wait fails at once without touching the GPU, and `onLost` lets the server restart the model process (the front
///   starts it again for the next request).
/// The wait itself stays Metal's own, with no completion handler: a buffer and what it binds go as soon as Metal is done
/// with it, as before. One watchdog thread looks at the open waits at their deadlines.
package enum GPUWait {
    /// Seconds a wait is allowed whatever its work, multiple of the expected time, and how many bounds a stuck buffer
    /// is then given to end. Settable for checks.
    nonisolated(unsafe) package static var floor = 10.0
    nonisolated(unsafe) package static var multiple = 20.0
    nonisolated(unsafe) package static var settle = 3.0
    /// Called once when a buffer did not end within its bound and `settle` more (CLI: exit so the front restarts).
    nonisolated(unsafe) package static var onLost: (@Sendable (String) -> Void)?
    /// Where the stall of a buffer is logged.
    nonisolated(unsafe) package static var log: (@Sendable (String) -> Void)?

    /// A check's fake: the next waited buffers' completions are seen this many seconds after they really end (nil:
    /// never), first to last; the real completion is always waited for too.
    package static func inject(_ delays: [Double?]) { state.withLock { $0.injected = delays } }
    /// Injected completions not yet used.
    package static var injectedLeft: Int { state.withLock { $0.injected.count } }
    /// Whether a buffer never ended: this process's GPU objects may still be written.
    package static var lost: Bool { state.withLock { $0.lost } }
    /// Checks: the device is usable again.
    package static func recover() { state.withLock { $0 = State() } }

    package static func bound(work: Double, budget: CommandBudget?) -> Double {
        let limit = budget?.limit ?? CommandBudget.work
        return max(floor, multiple * max(0, work) * CommandBudget.seconds / limit)
    }

    /// Takes the device for lost (`gpu_lost`): later waits fail at once, and `onLost` runs the first time.
    package static func declareLost(_ message: String) -> EngineError {
        let first = state.withLock { state in
            defer { state.lost = true }
            return !state.lost
        }
        if first { onLost?(message) }
        return EngineError(message, code: "gpu_lost")
    }

    fileprivate struct State {
        var lost = false
        var injected: [Double?] = []
    }
    fileprivate static let state = Locked(State())
    package static let watchdog = Watchdog()

    /// The open waits' deadlines. Its thread sleeps until the earliest it knows of, then takes any wait past its
    /// deadline for lost; it goes idle only after such a look found none open, so back-to-back waits (each closed
    /// before the next opens) cost no wake-up of their own.
    package final class Watchdog: @unchecked Sendable {
        private let condition = NSCondition()
        private var open: [UInt64: (deadline: Date, message: String)] = [:]
        private var next: UInt64 = 0
        private var expired: Set<UInt64> = []
        private var started = false, idle = false
        /// The deadline the thread sleeps toward (nil: idle or looking).
        private var sleepingUntil: Date?
        package func register(deadline: Date, message: String) -> UInt64 {
            condition.lock()
            defer { condition.unlock() }
            next += 1
            open[next] = (deadline, message)
            if !started {
                started = true
                Thread.detachNewThread { [self] in run() }
            } else if idle || sleepingUntil.map({ deadline < $0 }) == true {
                // An earlier deadline than the one slept toward: look again now.
                condition.signal()
            }
            return next
        }
        /// Whether the watchdog took this wait's device for lost.
        package func close(_ id: UInt64) -> Bool {
            condition.lock()
            defer { condition.unlock() }
            open[id] = nil
            return expired.remove(id) != nil
        }
        private func run() {
            condition.lock()
            while true {
                // Waits already taken for lost stay open until their thread returns (perhaps never): not looked at again.
                guard let earliest = open.filter({ !expired.contains($0.key) }).values.map(\.deadline).min() else {
                    idle = true
                    condition.wait()
                    idle = false
                    continue
                }
                // Woken early (a wait with an earlier deadline opened, or spuriously): look again.
                if Date() < earliest {
                    sleepingUntil = earliest
                    _ = condition.wait(until: earliest)
                    sleepingUntil = nil
                    continue
                }
                let now = Date()
                for (id, wait) in open where wait.deadline <= now && !expired.contains(id) {
                    expired.insert(id)
                    condition.unlock()
                    _ = GPUWait.declareLost(wait.message)
                    condition.lock()
                }
            }
        }
    }
}

extension MTLCommandBuffer {
    /// `waitUntilCompleted`, bounded by `GPUWait.bound(work:budget:)` (`work`: multiply-adds committed on its queue since
    /// the last wait, it included). Callers check `status` as before; this throws only `gpu_stalled` or `gpu_lost`.
    package func waitBounded(work: Double = 0, budget: CommandBudget? = nil) throws {
        let name = label ?? "a GPU step"
        if GPUWait.lost { throw EngineError("The GPU stopped answering earlier (\(name) not run)", code: "gpu_lost") }
        let delay: Double?? = GPUWait.state.withLock { $0.injected.isEmpty ? .none : .some($0.injected.removeFirst()) }
        let bound = GPUWait.bound(work: work, budget: budget)
        let total = bound * (1 + GPUWait.settle)
        let start = Date()
        let lostMessage = String(format: "The GPU did not finish %@ in %.0f s", name, total)
        let id = GPUWait.watchdog.register(deadline: start + total, message: lostMessage)
        waitUntilCompleted()
        if GPUWait.watchdog.close(id) { throw EngineError(lostMessage, code: "gpu_lost") }
        var seen = Date()
        if case .some(let fake) = delay {
            // A fake late or lost completion (checks only): seen `fake` seconds after the real one, or never.
            seen = fake.map { seen + $0 } ?? .distantFuture
            Thread.sleep(until: min(seen, start + total))
        }
        let seconds = seen.timeIntervalSince(start)
        if seconds <= bound { return }
        guard seconds <= total else { throw GPUWait.declareLost(lostMessage) }
        GPUWait.log?(String(format: "GPU: %@ took %.1f s, over its bound of %.1f s; running the request again", name, seconds, bound))
        budget?.interrupted()
        throw EngineError(String(format: "The GPU took over %.1f s for %@", bound, name), code: "gpu_stalled")
    }
}
