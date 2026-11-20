import Dispatch
import Foundation
import OrigamiEngine

/// Standalone checks for `decode_ple_internal_race` (no XCTest): either source completes a
/// token's rows, packing takes each row from the source that won it, failures of one source are
/// absorbed and counted, both failing is an error, cancellation stops the join, the read-hygiene
/// hold is released at completion (not when the slower source stops), the default path is
/// unchanged, and the configuration and report keys follow the flag.
package enum DecodePLERaceSelfChecks {
    private typealias Check = DecodeFrontierSelfChecks
    private final class Gate: @unchecked Sendable {
        let semaphore = DispatchSemaphore(value: 0)
        func open(_ count: Int = 64) { for _ in 0..<count { semaphore.signal() } }
        func pass() { _ = semaphore.wait(timeout: .now() + 10) }
    }
    private struct Failure: Error {}
    /// Rows of one decode token: 16 indices with duplicates (12 unique).
    private static let rows = [7, 2_500_011, 2_500_012, 320_001_535, 7, 41, 99, 41, 1_000, 2_000, 3_000, 7, 4_000, 5_000, 6_000, 99]
    private static func bytes(_ row: Int, _ marker: UInt8) -> [UInt8] {
        (0..<PLERowPrefetch.rowBytes).map { UInt8(truncatingIfNeeded: row &* 31 &+ $0 &* 7) ^ marker }
    }
    private static func expected(_ marker: UInt8) -> [UInt8] { rows.flatMap { bytes($0, marker) } }
    private static func fill(_ row: Int, _ buffer: UnsafeMutableRawBufferPointer, _ marker: UInt8) {
        for (index, value) in bytes(row, marker).enumerated() { buffer[index] = value }
    }

    /// The slow source never decides a token: whichever source finishes a row first supplies its
    /// bytes, the late loser's bytes (marked differently here) never reach the packed rows, and
    /// the hold is released while the loser is still blocked.
    package static func eitherSourceWinsExactly() throws {
        let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
        let primaryPool = try SpanReadPool(threads: 16)
        let alternatePool = try SpanReadPool(threads: 16)
        let limits = IOScheduler.Limits(
            maximumConcurrent: 2, reservedDemandConcurrent: 1, maximumOutstanding: 8, reservedDemandOutstanding: 4,
            maximumOutstandingBytes: 32_768, reservedDemandBytes: 16_384)
        // The scheduler holds its own io-scheduler-metadata reservation for its lifetime, so it gets its
        // own ledger: `ledger` then carries only the race's device_ple_prefetched_rows reservations.
        let schedulerLedger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
        let scheduler = try IOScheduler(ledger: schedulerLedger, internalLimits: limits, externalLimits: limits, readHygiene: true)
        defer { scheduler.shutdown() }
        for slowPrimary in [true, false] {
            let gate = Gate()
            let fastMarker: UInt8 = 0
            let slowMarker: UInt8 = 0xEE
            guard let hold = scheduler.holdPrefetch() else { throw EngineError("PLE race fixture: no hold") }
            let slow: @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void = {
                gate.pass()
                fill($0, $1, slowMarker)
            }
            let fast: @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void = { fill($0, $1, fastMarker) }
            var prefetch: PLERowPrefetch? = try PLERowPrefetch(
                tokenRows: [rows], ledger: ledger, workers: 16, pool: primaryPool, hold: hold,
                race: (alternatePool, slowPrimary ? fast : slow), read: slowPrimary ? slow : fast)
            let waited = try prefetch!.wait(shouldContinue: { true })
            try Check.require(waited < 5 && prefetch!.raced && prefetch!.uniqueRows == 12, "Race join waited for the slow source")
            let counts = prefetch!.raceCounts()
            try Check.require(
                counts.primary == (slowPrimary ? 0 : 12) && counts.alternate == (slowPrimary ? 12 : 0) && counts.failures == 0,
                "Race winners differ (slow primary \(slowPrimary)): \(counts)")
            for _ in 0..<1_000 where scheduler.hygieneCounts()?.liveHolds != 0 { Thread.sleep(forTimeInterval: 0.001) }
            try Check.require(scheduler.hygieneCounts()?.liveHolds == 0, "Race kept the hold while the slow source was blocked")
            try Check.require(
                try prefetch!.packed(tokens: 0..<1) == expected(fastMarker), "Race packed rows differ before the loser finished")
            gate.open()
            Thread.sleep(forTimeInterval: 0.05)  // let the released losers write their own buffers
            try Check.require(
                try prefetch!.packed(tokens: 0..<1) == expected(fastMarker) && prefetch!.raceCounts().primary == counts.primary
                    && prefetch!.raceCounts().alternate == counts.alternate, "The late loser changed the packed rows or the winners")
            prefetch = nil
        }
        for _ in 0..<2_000 where ledger.snapshot().reserved > 0 { Thread.sleep(forTimeInterval: 0.001) }
        try Check.require(ledger.snapshot().reserved == 0, "PLE race fixture leaked its row reservation")
    }

    /// One failing source is absorbed (counted, must be zero in a run); both failing is an error;
    /// cancellation stops the join; the default path has no race state.
    package static func failuresCancellationAndDefault() throws {
        let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
        let primaryPool = try SpanReadPool(threads: 16)
        let alternatePool = try SpanReadPool(threads: 16)
        let good: @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void = { fill($0, $1, 0) }
        let bad: @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void = { _, _ in throw Failure() }
        do {
            // Primary fails every row before the alternate starts; the alternate completes the token.
            let gate = Gate()
            let late: @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void = {
                gate.pass()
                fill($0, $1, 0)
            }
            let prefetch = try PLERowPrefetch(
                tokenRows: [rows], ledger: ledger, workers: 16, pool: primaryPool,
                race: (alternatePool, late), read: bad)
            for _ in 0..<2_000 where prefetch.raceCounts().failures < 12 { Thread.sleep(forTimeInterval: 0.001) }
            try Check.require(prefetch.raceCounts().failures == 12, "Primary failures were not all counted")
            gate.open()
            _ = try prefetch.wait(shouldContinue: { true })
            let counts = prefetch.raceCounts()
            try Check.require(
                counts.primary == 0 && counts.alternate == 12 && counts.failures == 12
                    && (try prefetch.packed(tokens: 0..<1)) == expected(0), "One-source failure was not absorbed exactly: \(counts)")
        }
        do {
            let prefetch = try PLERowPrefetch(
                tokenRows: [rows], ledger: ledger, workers: 16, pool: primaryPool,
                race: (alternatePool, bad), read: bad)
            var threw = false
            do { _ = try prefetch.wait(shouldContinue: { true }) } catch is Failure { threw = true }
            try Check.require(threw, "Both sources failing did not fail the token")
        }
        do {
            let gate = Gate()
            let blocked: @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void = {
                gate.pass()
                fill($0, $1, 0)
            }
            let prefetch = try PLERowPrefetch(
                tokenRows: [rows], ledger: ledger, workers: 16, pool: primaryPool,
                race: (alternatePool, blocked), read: blocked)
            var cancelled = false
            do { _ = try prefetch.wait(shouldContinue: { false }) } catch is IOOperationCancelled { cancelled = true }
            gate.open()
            try Check.require(cancelled, "Race join ignored cancellation")
        }
        do {
            let prefetch = try PLERowPrefetch(tokenRows: [rows], ledger: ledger, workers: 16, pool: primaryPool, read: good)
            _ = try prefetch.wait(shouldContinue: { true })
            let counts = prefetch.raceCounts()
            try Check.require(
                !prefetch.raced && counts.primary == 0 && counts.alternate == 0 && counts.failures == 0
                    && (try prefetch.packed(tokens: 0..<1)) == expected(0), "Default PLE prefetch path changed")
        }
        for _ in 0..<2_000 where ledger.snapshot().reserved > 0 { Thread.sleep(forTimeInterval: 0.001) }
        try Check.require(ledger.snapshot().reserved == 0, "PLE race failure fixtures leaked their reservations")
    }

    /// Off by default and in an older trace configuration; requires device_ple and io_reader_pool; boolean only; report
    /// keys absent when off and present when set.
    package static func configurationAndReport() throws {
        try Check.require(!RuntimeConfiguration().decodePLEInternalRace, "PLE race defaults on")
        let base = CheckFixtures.file("config/trace-ple-race-8192.json")
        let traced = try RuntimeConfiguration.load(base)
        try Check.require(!traced.decodePLEInternalRace && traced.devicePLE && traced.ioReaderPool, "The PLE race trace fixture differs")
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-ple-race-" + UUID().uuidString + ".json")
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
        try Check.require(try load({ $0["decode_ple_internal_race"] = true }).decodePLEInternalRace, "Explicit true was not honoured")
        try Check.require(!(try load({ $0["decode_ple_internal_race"] = false }).decodePLEInternalRace), "Explicit false was not honoured")
        try refused(
            {
                $0["decode_ple_internal_race"] = true
                $0["io_reader_pool"] = false
            }, "the PLE race without io_reader_pool")
        try refused(
            {
                $0["decode_ple_internal_race"] = true
                $0["device_ple"] = false
            }, "the PLE race without device_ple")
        try refused({ $0["decode_ple_internal_race"] = 1 }, "a non-boolean PLE race flag")
        try refused({ $0["decode_ple_internal_race"] = "true" }, "a string PLE race flag")
    }
}
