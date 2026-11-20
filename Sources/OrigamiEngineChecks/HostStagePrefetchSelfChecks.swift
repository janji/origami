import Foundation
import OrigamiEngine

/// Standalone checks without original weights: embedding rows equal `ReferenceWeights.dequantizedRow`
/// on the same bytes (duplicates, a single row, pool and dedicated readers), read failures and
/// cancellation propagate, and the dense prefetcher keeps its distance bound and joins.
package enum HostStagePrefetchSelfChecks {
    private static func require(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
        if try !value() { throw EngineError(message) }
    }

    private static func rowBytes(_ id: Int, layout: EmbeddingRowPrefetch.Layout) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: layout.rowBytes)
        for i in 0..<layout.wordBytes { bytes[i] = UInt8(truncatingIfNeeded: (UInt32(id * 1280 + i) &* 2_654_435_761) >> 11) }
        let groups = layout.scaleBytes / 2
        for g in 0..<groups {
            let scale = BF16.encode(Float((id + g) % 13 + 1) / 1024)
            let bias = BF16.encode(-Float((id * 7 + g) % 17) / 512)
            bytes[layout.wordBytes + g * 2] = UInt8(scale & 0xff)
            bytes[layout.wordBytes + g * 2 + 1] = UInt8(scale >> 8)
            bytes[layout.wordBytes + layout.scaleBytes + g * 2] = UInt8(bias & 0xff)
            bytes[layout.wordBytes + layout.scaleBytes + g * 2 + 1] = UInt8(bias >> 8)
        }
        // Exceptional scales: an infinite scale and a NaN bias on two fixed ids.
        if id == 7 {
            bytes[layout.wordBytes] = 0x80
            bytes[layout.wordBytes + 1] = 0x7f
        }
        if id == 11 {
            bytes[layout.wordBytes + layout.scaleBytes] = 0xc1
            bytes[layout.wordBytes + layout.scaleBytes + 1] = 0x7f
        }
        return bytes
    }

    private static func oracle(_ ids: [Int], layout: EmbeddingRowPrefetch.Layout) throws -> [Float] {
        var rows: [Int: [Float]] = [:]
        for id in Set(ids) {
            let data = Data(rowBytes(id, layout: layout))
            rows[id] = try ReferenceWeights.dequantizedRow(
                words: ReferenceWeights.array(data.subdata(in: 0..<layout.wordBytes), UInt32.self),
                scales: ReferenceWeights.array(data.subdata(in: layout.wordBytes..<(layout.wordBytes + layout.scaleBytes)), UInt16.self),
                biases: ReferenceWeights.array(data.subdata(in: (layout.wordBytes + layout.scaleBytes)..<layout.rowBytes), UInt16.self),
                groupSize: ModelLayout.group)
        }
        return ids.flatMap { rows[$0]! }
    }

    package static func exactness() throws {
        let ledger = try MemoryLedger(ceiling: 2_000_000_000, transientMargin: 1_000_000)
        let layout = EmbeddingRowPrefetch.Layout(wordBytes: 1280, scaleBytes: 80, biasBytes: 80)
        let read: @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void = { id, buffer in
            let bytes = rowBytes(id, layout: layout)
            guard buffer.count == bytes.count else { throw EngineError("Embedding fixture destination differs") }
            bytes.withUnsafeBytes { buffer.copyMemory(from: $0) }
        }
        let cases: [[Int]] = [
            [5], [3, 3, 3], Array(0..<40) + [7, 11, 7], (0..<3000).map { ($0 * 7919) % 1200 + ($0 % 3 == 0 ? 247_000 : 0) },
        ]
        for ids in cases {
            let prefetch = try EmbeddingRowPrefetch(ids: ids, layout: layout, ledger: ledger, read: read)
            _ = try prefetch.wait(shouldContinue: { true })
            let actual = try prefetch.embedding()
            let expected = try oracle(ids, layout: layout)
            try require(
                actual.count == ids.count * 2560 && zip(actual, expected).allSatisfy { $0.bitPattern == $1.bitPattern },
                "Embedding prefetch differs from the reference row dequantization (\(ids.count) ids)")
            try require(prefetch.uniqueRows == Set(ids).count, "Embedding prefetch unique row count differs")
        }
        // A failed read propagates from wait; cancellation surfaces as IOOperationCancelled.
        let failing = try EmbeddingRowPrefetch(ids: Array(0..<200), layout: layout, ledger: ledger) { id, buffer in
            if id == 150 { throw EngineError("fixture read failure") }
            try read(id, buffer)
        }
        do {
            _ = try failing.wait(shouldContinue: { true })
            throw EngineError("Embedding prefetch hid a read failure")
        } catch let error as EngineError where error.description.contains("fixture read failure") {}
        let slow = try EmbeddingRowPrefetch(ids: Array(0..<500), layout: layout, ledger: ledger) { id, buffer in
            usleep(2000)
            try read(id, buffer)
        }
        do {
            _ = try slow.wait(shouldContinue: { false })
            throw EngineError("Embedding prefetch ignored cancellation")
        } catch is IOOperationCancelled {}
        do {
            _ = try slow.embedding()
            throw EngineError("Embedding prefetch served unjoined rows")
        } catch is EngineError {}

        // Dense prefetcher: never more than `distance` layers ahead, every layer once, joins on finish.
        final class Log: @unchecked Sendable {
            let lock = NSLock()
            var entered = -1, order: [Int] = [], violations = 0
        }
        let log = Log()
        let dense = try DenseLayerPrefetch(layers: 48, distance: 2) { layer in
            log.lock.lock()
            defer { log.lock.unlock() }
            if layer > log.entered + 2 { log.violations += 1 }
            log.order.append(layer)
            return (reads: 3, complete: layer != 5)
        }
        for layer in 0..<48 {
            log.lock.lock()
            log.entered = layer
            log.lock.unlock()
            dense.enter(layer)
            usleep(200)
        }
        let result = dense.finish()
        try require(log.violations == 0, "Dense prefetch ran ahead of its distance bound")
        try require(log.order == Array(0..<48), "Dense prefetch skipped or reordered layers")
        try require(
            result.layers == 48 && result.reads == 144 && result.incomplete == 1 && result.failures == 0,
            "Dense prefetch counters differ")
        // An early finish drains to the bound: none entered with distance 1 allows layer 0 only,
        // entering layer 9 with distance 2 allows 0...11. A throwing layer stops and is counted.
        let early = try DenseLayerPrefetch(layers: 48, distance: 1) { _ in (reads: 1, complete: true) }.finish()
        try require(early.layers == 1 && early.failures == 0, "Dense prefetch did not stop at its bound")
        let partial = try DenseLayerPrefetch(layers: 48, distance: 2) { _ in (reads: 1, complete: true) }
        partial.enter(9)
        let drained = partial.finish()
        try require(drained.layers == 12 && drained.reads == 12, "Dense prefetch did not drain to its bound")
        let throwing = try DenseLayerPrefetch(layers: 4, distance: 8) { layer in
            if layer == 2 { throw EngineError("fixture dense failure") }
            return (reads: 1, complete: true)
        }
        let failed = throwing.finish()
        try require(failed.layers == 2 && failed.failures == 1, "Dense prefetch failure accounting differs")
    }
}
