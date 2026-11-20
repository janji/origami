import Foundation
import OrigamiEngine

/// Standalone checks: disjoint writes from concurrent callers land exactly, every task has stopped
/// before `run` returns, the first error propagates only after all tasks finished, and a stopped
/// pool refuses `run` while still running what was queued.
package enum SpanReadPoolSelfChecks {
    package static func exactness() throws {
        let pool = try SpanReadPool(threads: 4)
        final class Box: @unchecked Sendable {
            var bytes = [UInt8](repeating: 0, count: 9 * 4096)
            let lock = NSLock()
            var finished = 0
        }
        final class Failures: @unchecked Sendable {
            private let lock = NSLock()
            private(set) var all: [String] = []
            func add(_ line: String) {
                lock.lock()
                all.append(line)
                lock.unlock()
            }
        }
        let callers = 12
        let failed = Failures()
        DispatchQueue.concurrentPerform(iterations: callers) { caller in
            let box = Box()
            do {
                try pool.run(
                    (0..<9).map { part in
                        {
                            usleep(UInt32(200 * (part % 3)))
                            box.lock.lock()
                            for i in (part * 4096)..<((part + 1) * 4096) {
                                box.bytes[i] = UInt8(truncatingIfNeeded: caller &* 31 &+ part &+ i)
                            }
                            box.finished += 1
                            box.lock.unlock()
                        }
                    })
                box.lock.lock()
                defer { box.lock.unlock() }
                let exact =
                    box.finished == 9
                    && box.bytes.indices.allSatisfy { box.bytes[$0] == UInt8(truncatingIfNeeded: caller &* 31 &+ $0 / 4096 &+ $0) }
                if !exact { failed.add("caller \(caller) bytes") }
            } catch { failed.add("caller \(caller): \(error)") }
        }
        guard failed.all.isEmpty else { throw EngineError("Span read pool lost or misplaced writes: \(failed.all.prefix(3))") }
        let box = Box()
        do {
            try pool.run(
                (0..<9).map { part in
                    {
                        if part == 4 { throw EngineError("fixture span failure") }
                        usleep(2000)
                        box.lock.lock()
                        box.finished += 1
                        box.lock.unlock()
                    }
                })
            throw EngineError("Span read pool hid a failed span")
        } catch let error as EngineError where error.description.contains("fixture span failure") {}
        box.lock.lock()
        let finished = box.finished
        box.lock.unlock()
        guard finished == 8 else { throw EngineError("Span read pool returned before its other spans stopped: \(finished)") }
        // Lifetime: a stopped pool runs what was queued, then refuses new runs; enqueue falls back inline.
        let queued = DispatchGroup()
        let late = Box()
        for _ in 0..<4 {
            queued.enter()
            pool.enqueue {
                usleep(1000)
                late.lock.lock()
                late.finished += 1
                late.lock.unlock()
                queued.leave()
            }
        }
        pool.stop()
        guard queued.wait(timeout: .now() + .seconds(10)) == .success else {
            throw EngineError("Stopped span read pool dropped queued tasks")
        }
        guard pool.isStopped else { throw EngineError("Span read pool did not record its stop") }
        do {
            try pool.run([{}])
            throw EngineError("Stopped span read pool accepted a run")
        } catch let error as EngineError where error.description.contains("is stopped") {}
        queued.enter()
        pool.enqueue {
            late.lock.lock()
            late.finished += 1
            late.lock.unlock()
            queued.leave()
        }
        guard queued.wait(timeout: .now() + .seconds(10)) == .success else {
            throw EngineError("Stopped span read pool lost an inline task")
        }
        late.lock.lock()
        let lateFinished = late.finished
        late.lock.unlock()
        guard lateFinished == 5 else { throw EngineError("Stopped span read pool task count differs: \(lateFinished)") }
    }
}
