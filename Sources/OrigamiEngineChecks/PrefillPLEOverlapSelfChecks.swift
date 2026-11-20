import Dispatch
import Foundation
import OrigamiEngine

/// Standalone checks for `prefill_ple_overlap` (no XCTest): a scope-sized race with the alternate
/// source descending yields exactly the bytes of either source, both sources contribute rows, one
/// slow source never decides the join, `readsComplete` is non-blocking and agrees with the join,
/// `copyRow` returns the winner's bytes only after the join, the host tail-tile row source serves
/// scope rows and counts misses, the default (ascending, unraced) paths are unchanged, and the
/// configuration and report keys follow the flag.
package enum PrefillPLEOverlapSelfChecks {
    private typealias Check = DecodeFrontierSelfChecks
    private final class Gate: @unchecked Sendable {
        let semaphore = DispatchSemaphore(value: 0)
        func open(_ count: Int = 256) { for _ in 0..<count { semaphore.signal() } }
        func pass() { _ = semaphore.wait(timeout: .now() + 10) }
    }
    /// A 40-token scope: 16 rows per token with repeats across tokens (n-gram rows recur).
    private static let scope: [[Int]] = (0..<40).map { token in
        (0..<16).map { head in head < 4 ? head * 1_000 + token % 5 : (token * 16 + head) * 7_919 % 320_001_536 }
    }
    private static func bytes(_ row: Int, _ marker: UInt8) -> [UInt8] {
        (0..<PLERowPrefetch.rowBytes).map { UInt8(truncatingIfNeeded: row &* 31 &+ $0 &* 7) ^ marker }
    }
    private static func expected(_ tokens: Range<Int>) -> [UInt8] { scope[tokens].joined().flatMap { bytes($0, 0) } }
    private static func fill(_ row: Int, _ buffer: UnsafeMutableRawBufferPointer, _ marker: UInt8) {
        for (index, value) in bytes(row, marker).enumerated() { buffer[index] = value }
    }

    /// Both sources at speed: disjoint cursors make both win rows; exact packing for every token range.
    /// One source blocked: the other completes the scope alone, the join does not wait for the blocked
    /// one, and its late bytes (marked) never reach `packed` or `copyRow`.
    package static func scopeRaceExactAndDisjoint() throws {
        let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
        let alternatePool = try SpanReadPool(threads: 32)
        let unique = Set(scope.joined()).count
        do {
            let delay: @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void = {
                usleep(200)
                fill($0, $1, 0)
            }
            let prefetch = try PLERowPrefetch(
                tokenRows: scope, ledger: ledger, workers: 8, race: (alternatePool, delay),
                alternateDescending: true, read: delay)
            _ = try prefetch.wait(shouldContinue: { true })
            let counts = prefetch.raceCounts()
            try Check.require(
                prefetch.raced && prefetch.uniqueRows == unique && counts.primary + counts.alternate == unique
                    && counts.primary > 0 && counts.alternate > 0 && counts.failures == 0,
                "Descending scope race did not split rows between both sources: \(counts)")
            try Check.require(
                try prefetch.packed(tokens: 0..<40) == expected(0..<40)
                    && (try prefetch.packed(tokens: 13..<29)) == expected(13..<29), "Scope race packed rows differ")
        }
        for slowPrimary in [true, false] {
            let gate = Gate()
            let slow: @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void = {
                gate.pass()
                fill($0, $1, 0xEE)
            }
            let fast: @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void = { fill($0, $1, 0) }
            let prefetch = try PLERowPrefetch(
                tokenRows: scope, ledger: ledger, workers: 8, race: (alternatePool, slowPrimary ? fast : slow),
                alternateDescending: true, read: slowPrimary ? slow : fast)
            let waited = try prefetch.wait(shouldContinue: { true })
            let counts = prefetch.raceCounts()
            try Check.require(
                waited < 5 && counts.failures == 0 && (slowPrimary ? counts.alternate : counts.primary) == unique,
                "Scope race waited for the blocked source (slow primary \(slowPrimary)): \(counts)")
            gate.open()
            Thread.sleep(forTimeInterval: 0.05)
            var row = [UInt8](repeating: 0, count: PLERowPrefetch.rowBytes)
            let first = scope[0][5]
            let found = try row.withUnsafeMutableBytes { try prefetch.copyRow(first, into: $0) }
            try Check.require(
                found && row == bytes(first, 0) && (try prefetch.packed(tokens: 0..<40)) == expected(0..<40),
                "A late loser's bytes reached the scope rows (slow primary \(slowPrimary))")
        }
        for _ in 0..<2_000 where ledger.snapshot().reserved > 0 { Thread.sleep(forTimeInterval: 0.001) }
        try Check.require(ledger.snapshot().reserved == 0, "Scope race fixture leaked its row reservation")
    }

    /// `readsComplete` never blocks and is true exactly when a join would not wait; `copyRow` refuses before
    /// the join and returns false for rows outside the scope; the tail row source serves scope rows and
    /// reads (and counts) only a row the scope lacks; the unraced default path is unchanged.
    package static func joinPollAndTailRows() throws {
        let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
        let alternatePool = try SpanReadPool(threads: 32)
        for raced in [true, false] {
            let gate = Gate()
            let blocked: @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void = {
                gate.pass()
                fill($0, $1, 0)
            }
            let prefetch = try PLERowPrefetch(
                tokenRows: scope, ledger: ledger, workers: 8,
                race: raced ? (alternatePool, blocked) : nil, alternateDescending: raced, read: blocked)
            let start = DispatchTime.now().uptimeNanoseconds
            let early = prefetch.readsComplete
            try Check.require(!early && DispatchTime.now().uptimeNanoseconds - start < 50_000_000, "readsComplete blocked or was early")
            var row = [UInt8](repeating: 0, count: PLERowPrefetch.rowBytes)
            var refused = false
            do { _ = try row.withUnsafeMutableBytes { try prefetch.copyRow(scope[0][0], into: $0) } } catch is EngineError {
                refused = true
            }
            try Check.require(refused, "copyRow served rows before the join")
            gate.open(4_096)
            for _ in 0..<5_000 where !prefetch.readsComplete { Thread.sleep(forTimeInterval: 0.001) }
            try Check.require(prefetch.readsComplete, "readsComplete never became true")
            let waited = try prefetch.wait(shouldContinue: { true })
            try Check.require(
                waited < 0.5 && prefetch.isJoined && prefetch.raced == raced
                    && (try prefetch.packed(tokens: 0..<40)) == expected(0..<40), "Join after a complete poll waited or differs")

            let storeReads = Counter()
            let source = try PLEScopeRowSource(prefetch: prefetch) { row, buffer in
                storeReads.add()
                fill(row, buffer, 0)
            }
            let reader = source.reader
            for index in Set(scope[30..<40].joined()) {
                var out = [UInt8](repeating: 0, count: PLERowPrefetch.rowBytes)
                try out.withUnsafeMutableBytes { try reader(index, $0) }
                try Check.require(out == bytes(index, 0), "Tail row source bytes differ at \(index)")
            }
            let outside = 319_999_999
            let servedOutside = try row.withUnsafeMutableBytes { try prefetch.copyRow(outside, into: $0) }
            try Check.require(!servedOutside, "copyRow served a row outside the scope")
            var out = [UInt8](repeating: 0, count: PLERowPrefetch.rowBytes)
            try out.withUnsafeMutableBytes { try reader(outside, $0) }
            let counts = source.counts()
            try Check.require(
                counts.served == UInt64(Set(scope[30..<40].joined()).count) && counts.missed == 1 && storeReads.value == 1
                    && out == bytes(outside, 0), "Tail row source counts differ: \(counts), store reads \(storeReads.value)")
        }
        do {
            let unjoined = try PLERowPrefetch(tokenRows: scope, ledger: ledger, workers: 8) { fill($0, $1, 0) }
            var refused = false
            do { _ = try PLEScopeRowSource(prefetch: unjoined) { _, _ in } } catch is EngineError { refused = true }
            try Check.require(refused, "Tail row source accepted an unjoined prefetch")
            _ = try unjoined.wait(shouldContinue: { true })
        }
        for _ in 0..<2_000 where ledger.snapshot().reserved > 0 { Thread.sleep(forTimeInterval: 0.001) }
        try Check.require(ledger.snapshot().reserved == 0, "Join/tail fixtures leaked their reservations")
    }

    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        func add() {
            lock.lock()
            count += 1
            lock.unlock()
        }
        var value: Int {
            lock.lock()
            defer { lock.unlock() }
            return count
        }
    }

    /// Off by default and in the trace fixture configuration; requires device_ple; boolean only; report keys absent when off and
    /// present when set.
    package static func configurationAndReport() throws {
        try Check.require(!RuntimeConfiguration().prefillPLEOverlap, "prefill_ple_overlap defaults on")
        let base = CheckFixtures.file("config/trace-prefill-ple-overlap-8192.json")
        let traced = try RuntimeConfiguration.load(base)
        try Check.require(!traced.prefillPLEOverlap && traced.devicePLE, "The prefill PLE overlap trace fixture differs")
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-ple-overlap-" + UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: url) }
        let original = try JSONSerialization.jsonObject(with: Data(contentsOf: base)) as! [String: Any]
        func load(_ mutate: (inout [String: Any]) -> Void) throws -> RuntimeConfiguration {
            var object = original
            var execution = original["execution"] as! [String: Any]
            mutate(&execution)
            object["execution"] = execution
            try JSONSerialization.data(withJSONObject: object).write(to: url)
            return try RuntimeConfiguration.load(url)
        }
        func refused(_ mutate: (inout [String: Any]) -> Void, _ label: String) throws {
            do { _ = try load(mutate) } catch is EngineError { return }
            throw EngineError("Configuration accepted " + label)
        }
        try Check.require(try load({ $0["prefill_ple_overlap"] = true }).prefillPLEOverlap, "Explicit true was not honoured")
        try Check.require(!(try load({ $0["prefill_ple_overlap"] = false }).prefillPLEOverlap), "Explicit false was not honoured")
        try refused(
            {
                $0["prefill_ple_overlap"] = true
                $0["device_ple"] = false
                $0["decode_ple_internal_race"] = false
            },
            "prefill_ple_overlap without device_ple")
        try refused({ $0["prefill_ple_overlap"] = 1 }, "a non-boolean prefill_ple_overlap flag")
        try refused({ $0["prefill_ple_overlap"] = "true" }, "a string prefill_ple_overlap flag")
    }
}
