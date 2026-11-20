import Foundation
import OrigamiEngine

private func demand(_ value: Bool, _ message: String) throws {
    guard value else { throw EngineError(message) }
}
private final class Writers: @unchecked Sendable {
    let condition = NSCondition()
    var entered = 0, completed = 0, active = 0
    func begin() {
        condition.lock()
        entered += 1
        active += 1
        condition.broadcast()
        while entered < 8 { condition.wait() }
        condition.unlock()
    }
    func end() {
        condition.lock()
        active -= 1
        completed += 1
        condition.unlock()
    }
    func canContinue() -> Bool {
        condition.lock()
        defer { condition.unlock() }
        return entered < 8
    }
    func snapshot() -> (Int, Int, Int) {
        condition.lock()
        defer { condition.unlock() }
        return (entered, completed, active)
    }
}

func checkKnownPLERows() throws {
    let ledger = try MemoryLedger(ceiling: 100 * 1024 * 1024, transientMargin: 1)
    func awaitRetirement() throws {
        // DispatchGroup completion can precede callback ARC cleanup. Each
        // case gets the same one-batch budget only after the prior owner has
        // actually retired; a leaked charge still fails this bounded wait.
        let until = Date().addingTimeInterval(1)
        while ledger.snapshot().reserved != 0 && Date() < until { Thread.sleep(forTimeInterval: 0.001) }
        try demand(ledger.snapshot().reserved == 0, "Known-row owner leaked admission")
    }
    let stream = [1, 1, 2, 3, 1, 4, 2, 5, 4, 2, 1, 1, 3, 5, 2, 4]
    var expected = [Int]()
    var cache = Set<Int>()
    var fifo = [Int](repeating: -1, count: 3)
    var next = 0
    for index in stream {
        if !cache.contains(index) {
            expected.append(index)
            cache.remove(fifo[next])
            cache.insert(index)
            fifo[next] = index
            next = (next + 1) % 3
        }
    }
    try autoreleasepool {
        let batch = try KnownPLERows(ids: [stream], fifo: [-1, -1, -1], next: 0, ledger: ledger)
        try demand(batch.indices == expected, "FIFO duplicate/eviction/revisit plan differs")
        try batch.load(shouldContinue: { true }) { index, bytes in
            bytes.initializeMemory(as: UInt8.self, repeating: UInt8(index))
        }
        for index in expected {
            try demand(try batch.consume(index) == Data(repeating: UInt8(index), count: 100), "Disjoint known-row payload changed")
        }
        try batch.finishConsumption()
        do {
            _ = try batch.consume(1)
            throw EngineError("Overconsumption accepted")
        } catch let error as EngineError { try demand(error.description.contains("replay differs"), "Unexpected replay refusal") }
    }
    try awaitRetirement()
    try autoreleasepool {
        let batch = try KnownPLERows(ids: [Array(repeating: 2, count: 16)], fifo: [1, 2, 3], next: 1, ledger: ledger)
        try demand(batch.indices.isEmpty, "Resident duplicates were reread")
        try batch.load(shouldContinue: { true }) { _, _ in throw EngineError("All-hit plan performed I/O") }
        try batch.finishConsumption()
    }
    try awaitRetirement()
    let denied = try MemoryLedger(ceiling: KnownPLERows.admittedBytes, transientMargin: 1)
    do {
        _ = try KnownPLERows(ids: [stream], fifo: [-1], next: 0, ledger: denied)
        throw EngineError("Admission unexpectedly passed")
    } catch let error as EngineError { try demand(error.code == "insufficient_memory", "Incorrect memory refusal") }
    try demand(denied.snapshot().reserved == 0, "Refused batch retained admission")
    for cancel in [true, false] {
        try autoreleasepool {
            let writers = Writers()
            let batch = try KnownPLERows(ids: [Array(0..<16)], fifo: [-1], next: 0, ledger: ledger)
            var rejected = false
            do {
                try batch.load(shouldContinue: { !cancel || writers.canContinue() }) { index, bytes in
                    writers.begin()
                    defer { writers.end() }
                    if !cancel && index == 0 { throw EngineError("Injected original read failure") }
                    Thread.sleep(forTimeInterval: 0.02)
                    bytes.initializeMemory(as: UInt8.self, repeating: UInt8(index))
                }
            } catch is IOOperationCancelled { rejected = cancel } catch let error as EngineError {
                rejected = !cancel && error.description == "Injected original read failure"
            }
            let counts = writers.snapshot()
            try demand(
                rejected && counts.0 == 8 && counts.1 == 8 && counts.2 == 0,
                "Failure/cancellation escaped before every writer joined or dispatched extra reads")
            do {
                _ = try batch.consume(0)
                throw EngineError("Failed batch published")
            } catch let error as EngineError { try demand(error.description.contains("replay differs"), "Failed bytes escaped") }
        }
        try awaitRetirement()
    }
    try autoreleasepool {
        let batch = try KnownPLERows(ids: [stream], fifo: [-1], next: 0, ledger: ledger)
        do {
            try batch.load(shouldContinue: { false }) { _, _ in throw EngineError("Pre-cancel dispatched I/O") }
            throw EngineError("Pre-cancel accepted")
        } catch is IOOperationCancelled {}
    }
    try awaitRetirement()
}
