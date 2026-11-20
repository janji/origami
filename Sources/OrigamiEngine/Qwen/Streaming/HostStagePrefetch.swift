import Foundation

/// Unique embedding rows of one prefill scope, read concurrently before the expert read-ahead starts
/// (the device PLE lesson: concurrent multi-megabyte expert reads slow small row reads 3-6x).
/// Each row is the checkpoint's affine4 row: words, then scales, then biases, exactly as
/// `ReferenceWeights.embedding` reads them; `embedding` dequantizes with the same function.
package final class EmbeddingRowPrefetch: @unchecked Sendable {
    /// Bytes of one embedding row's three parts as stored in the checkpoint (u32 code words, bf16 scales, bf16 biases).
    package struct Layout: Equatable {
        package let wordBytes, scaleBytes, biasBytes: Int
        package init(wordBytes: Int, scaleBytes: Int, biasBytes: Int) {
            self.wordBytes = wordBytes
            self.scaleBytes = scaleBytes
            self.biasBytes = biasBytes
        }
        package var rowBytes: Int { wordBytes + scaleBytes + biasBytes }
    }
    private static let maximumStack = 8 * 1024 * 1024
    private static let workerStack = 512 * 1024, dedicatedThreshold = 64
    let ids: [Int]
    let layout: Layout
    private let unique: [Int]
    private let slots: [Int: Int]
    private let payload: UnsafeMutableRawPointer
    private let ledger: MemoryLedger, reservation: MemoryLedger.Reservation
    private let lock = NSLock()
    private let group = DispatchGroup()
    private var cursor = 0, stopped = false, failure: Error?, joined = false

    package init(
        ids: [Int], layout: Layout, ledger: MemoryLedger, workers: Int = 32,
        read: @escaping @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void
    ) throws {
        guard !ids.isEmpty, ids.allSatisfy({ (0..<248_320).contains($0) }), (1...64).contains(workers),
            layout.wordBytes > 0, layout.wordBytes % 4 == 0, layout.scaleBytes == layout.biasBytes,
            layout.scaleBytes > 0, layout.scaleBytes % 2 == 0,
            (layout.wordBytes / 4 * 8) % (layout.scaleBytes / 2) == 0
        else {
            throw EngineError("Embedding row prefetch geometry differs")
        }
        var slots: [Int: Int] = [:]
        var unique: [Int] = []
        for id in ids where slots[id] == nil {
            slots[id] = unique.count
            unique.append(id)
        }
        let bytes = UInt64(unique.count * layout.rowBytes + unique.count * 32 + ids.count * 8)
        reservation = try ledger.reserve(bytes: bytes, category: "embedding_prefetched_rows")
        self.ledger = ledger
        self.ids = ids
        self.layout = layout
        self.unique = unique
        self.slots = slots
        payload = .allocate(byteCount: max(1, unique.count * layout.rowBytes), alignment: 16)
        // Scope-sized reads use dedicated threads, like PLERowPrefetch: GCD's pool does not provide
        // `workers` concurrent blocking preads. A handful of rows keeps the pool.
        let dedicated = unique.count > Self.dedicatedThreshold
        let rowBytes = layout.rowBytes
        for _ in 0..<min(workers, unique.count) {
            group.enter()
            let work: @Sendable () -> Void = { [self] in
                defer { group.leave() }
                do {
                    guard pthread_get_stacksize_np(pthread_self()) <= Self.maximumStack else {
                        throw EngineError("Embedding prefetch worker exceeds admitted stack bound")
                    }
                    while let slot = next() {
                        try read(
                            self.unique[slot], UnsafeMutableRawBufferPointer(start: payload.advanced(by: slot * rowBytes), count: rowBytes))
                    }
                } catch { fail(error) }
            }
            if dedicated {
                let thread = Thread(block: work)
                thread.stackSize = Self.workerStack
                thread.qualityOfService = .userInitiated
                thread.start()
            } else {
                DispatchQueue.global(qos: .userInitiated).async(execute: work)
            }
        }
    }

    deinit {
        cancel()
        group.wait()
        payload.deallocate()
        try? ledger.release(reservation)
    }

    package var uniqueRows: Int { unique.count }

    private func next() -> Int? {
        lock.lock()
        defer { lock.unlock() }
        guard !stopped, cursor < unique.count else { return nil }
        defer { cursor += 1 }
        return cursor
    }
    private func fail(_ error: Error) {
        lock.lock()
        defer { lock.unlock() }
        stopped = true
        if failure == nil { failure = error }
    }
    func cancel() {
        lock.lock()
        stopped = true
        lock.unlock()
    }

    /// Join all readers (polling the caller's cancellation) and return the seconds spent waiting; a read that does not
    /// return within the demand read bound fails with `drive_stalled` (`ExpertReadStall.join`; the readers keep `self`).
    package func wait(shouldContinue: () -> Bool) throws -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        if !joined {
            try ExpertReadStall.join(
                group, bytes: unique.count * layout.rowBytes, what: "embedding row", shouldContinue: shouldContinue, cancel: cancel)
            joined = true
        }
        lock.lock()
        let error = failure
        let complete = cursor >= unique.count
        lock.unlock()
        if let error { throw error }
        guard complete else { throw IOOperationCancelled() }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
    }

    /// The scope's embeddings in token order, identical to `ReferenceWeights.embedding(ids)`.
    package func embedding() throws -> [Float] {
        guard joined else { throw EngineError("Embedding prefetch used before its readers joined") }
        let rowBytes = layout.rowBytes
        let groupSize = layout.wordBytes / 4 * 8 / (layout.scaleBytes / 2)
        var rows: [[Float]] = []
        rows.reserveCapacity(unique.count)
        for slot in unique.indices {
            let row = Data(bytes: payload.advanced(by: slot * rowBytes), count: rowBytes)
            rows.append(
                try ReferenceWeights.dequantizedRow(
                    words: ReferenceWeights.array(row.subdata(in: 0..<layout.wordBytes), UInt32.self),
                    scales: ReferenceWeights.array(row.subdata(in: layout.wordBytes..<(layout.wordBytes + layout.scaleBytes)), UInt16.self),
                    biases: ReferenceWeights.array(row.subdata(in: (layout.wordBytes + layout.scaleBytes)..<rowBytes), UInt16.self),
                    groupSize: groupSize))
        }
        return ids.flatMap { rows[slots[$0]!] }
    }
}

/// Warms the resident dense caches for prefill layers 0..<layers on one background thread, at most
/// `distance` layers ahead of the layer the loop has entered, so the reads overlap the GPU work of
/// earlier layers instead of stalling `acquireWholeLayerWeights`. It only admits what the loop would
/// admit itself, never evicts, and holds no lease: the loop's acquisition stays the authority for
/// validation, geometry and errors. A failed or partial layer is counted and left to the loop.
package final class DenseLayerPrefetch: @unchecked Sendable {
    private let condition = NSCondition()
    private let group = DispatchGroup()
    private var entered = -1, draining = false
    private var layers = 0, reads = 0, incomplete = 0, failures = 0

    /// `prefetch(layer)` returns (tensor reads, complete).
    package init(layers count: Int, distance: Int, prefetch: @escaping @Sendable (Int) throws -> (reads: Int, complete: Bool)) throws {
        guard (1...48).contains(count), (0...8).contains(distance) else { throw EngineError("Dense layer prefetch bounds differ") }
        group.enter()
        let thread = Thread { [self] in
            defer { group.leave() }
            for layer in 0..<count {
                condition.lock()
                while !draining && layer > entered + distance { condition.wait() }
                let stop = layer > entered + distance
                condition.unlock()
                if stop { break }
                do {
                    let result = try prefetch(layer)
                    condition.lock()
                    layers += 1
                    reads += result.reads
                    incomplete += result.complete ? 0 : 1
                    condition.unlock()
                } catch {
                    condition.lock()
                    failures += 1
                    condition.unlock()
                    break
                }
            }
        }
        thread.stackSize = 512 * 1024
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    deinit { _ = finish() }

    /// The loop has entered `layer`; the prefetcher may run up to `layer + distance`.
    package func enter(_ layer: Int) {
        condition.lock()
        entered = max(entered, layer)
        condition.broadcast()
        condition.unlock()
    }

    /// Complete every layer up to the bound of the last entered layer (all of them after a full
    /// loop; at most `distance` more after an early exit), then join the thread.
    package func finish() -> (layers: Int, reads: Int, incomplete: Int, failures: Int) {
        condition.lock()
        draining = true
        condition.broadcast()
        condition.unlock()
        group.wait()
        condition.lock()
        defer { condition.unlock() }
        return (layers, reads, incomplete, failures)
    }
}

/// Sendable handle for the dense prefetcher thread, which calls only the lock-protected
/// resident cache methods behind `ReferenceWeights.prefetchWholeLayer`.
final class HostStageWeights: @unchecked Sendable {
    let weights: ReferenceWeights
    init(_ weights: ReferenceWeights) { self.weights = weights }
}
