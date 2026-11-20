import Darwin
import Foundation

/// Concurrent, joined reads of the 16 n-gram PLE rows per token for one prefill scope or one decode
/// token. Rows are deduplicated and read through the caller's exact row reader (PackStore identity
/// checks unchanged); bytes are identical to ReferencePLE's, only their timing differs.
package final class PLERowPrefetch: @unchecked Sendable {
    package static let rowBytes = 100, rowsPerToken = 16
    private static let maximumStack = 8 * 1024 * 1024
    private static let workerStack = 512 * 1024, dedicatedThreshold = 64
    let tokenRows: [[Int]]
    private let unique: [Int]
    private let slots: [Int: Int]
    private let payload: UnsafeMutableRawPointer
    /// `decode_ple_internal_race` (nil when off): a second, independent source for every row. Each
    /// source has its own workers, cursor and payload; a row is complete when either source has read
    /// it, and readers skip rows the other source has already won. Bytes are identical by construction
    /// (the packed row is the concatenation of the three original spans).
    private let alternate: UnsafeMutableRawPointer?
    private var winner: [Int8] = [], failedSources: [UInt8] = []
    private var alternateCursor = 0, wonRows = 0, finished = false
    /// `prefill_ple_overlap` scope races: the alternate source walks the rows from the last one down,
    /// so the two sources read disjoint rows until their cursors cross (decode tokens keep both ascending).
    private let alternateDescending: Bool
    private var primaryWins = 0, alternateWins = 0, sourceFailures = 0
    private let complete = DispatchGroup()
    private let raceHold: IOPrefetchHold?
    /// Without a race: released when every reader stopped, or when the wait gives up on a stalled read.
    private let readHold: IOPrefetchHold?
    private let ledger: MemoryLedger, reservation: MemoryLedger.Reservation
    private let lock = NSLock()
    private let group = DispatchGroup()
    private var cursor = 0, stopped = false, failure: Error?, joined = false
    private let began = DispatchTime.now().uptimeNanoseconds

    /// `hold` (read hygiene, decode only): released once every row reader has stopped, so no
    /// predicted expert read starts while this token's rows are being read.
    /// `race`: the alternate reader and its own pool; `hold` is then released once every
    /// row has a winner, not when the slower source's readers stop. `alternateDescending` (scope races
    /// only) makes the alternate source start at the last row, so both sources add throughput.
    package init(
        tokenRows: [[Int]], ledger: MemoryLedger, workers: Int = 32, pool: SpanReadPool? = nil, hold: IOPrefetchHold? = nil,
        race: (pool: SpanReadPool, read: @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void)? = nil,
        alternateDescending: Bool = false,
        read: @escaping @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void
    ) throws {
        guard !tokenRows.isEmpty, tokenRows.allSatisfy({ $0.count == Self.rowsPerToken }),
            tokenRows.joined().allSatisfy({ (0..<320_001_536).contains($0) }), (1...64).contains(workers)
        else {
            throw EngineError("PLE row prefetch geometry differs")
        }
        var slots: [Int: Int] = [:]
        var unique: [Int] = []
        for index in tokenRows.joined() where slots[index] == nil {
            slots[index] = unique.count
            unique.append(index)
        }
        let raced = race != nil
        let descending = raced && alternateDescending
        self.alternateDescending = descending
        let bytes = UInt64(
            unique.count * Self.rowBytes * (raced ? 2 : 1) + unique.count * (raced ? 34 : 32)
                + tokenRows.count * Self.rowsPerToken * 8)
        reservation = try ledger.reserve(bytes: bytes, category: "device_ple_prefetched_rows")
        self.ledger = ledger
        self.tokenRows = tokenRows
        self.unique = unique
        self.slots = slots
        payload = .allocate(byteCount: max(1, unique.count * Self.rowBytes), alignment: 16)
        if let race {
            alternate = .allocate(byteCount: max(1, unique.count * Self.rowBytes), alignment: 16)
            winner = [Int8](repeating: -1, count: unique.count)
            failedSources = [UInt8](repeating: 0, count: unique.count)
            raceHold = hold
            readHold = nil
            alternateCursor = descending ? unique.count - 1 : 0
            complete.enter()
            for _ in 0..<min(workers, unique.count) {
                group.enter()
                let work: @Sendable () -> Void = { [self] in
                    defer { group.leave() }
                    raceWorker(source: 1, read: race.read)
                }
                race.pool.enqueue(work)
            }
        } else {
            alternate = nil
            raceHold = nil
            readHold = hold
        }
        // Scope-sized reads use dedicated threads: GCD's pool does not provide `workers` concurrent blocking
        // preads (timed 8K: about 13.5k rows/s; the drive gives about 47k at 32 threads). A decode token's
        // 16 rows keep the pool, where spawning threads would cost more than it saves.
        let dedicated = unique.count > Self.dedicatedThreshold
        for _ in 0..<min(workers, unique.count) {
            group.enter()
            let work: @Sendable () -> Void = { [self] in
                defer { group.leave() }
                do {
                    guard pthread_get_stacksize_np(pthread_self()) <= Self.maximumStack else {
                        throw EngineError("PLE prefetch worker exceeds admitted stack bound")
                    }
                    if raced {
                        raceWorker(source: 0, read: read)
                        return
                    }
                    while let slot = next() {
                        try read(
                            self.unique[slot],
                            UnsafeMutableRawBufferPointer(
                                start: payload.advanced(by: slot * Self.rowBytes),
                                count: Self.rowBytes))
                    }
                } catch { fail(error) }
            }
            if let pool {
                pool.enqueue(work)
            } else if dedicated {
                let thread = Thread(block: work)
                thread.stackSize = Self.workerStack
                thread.qualityOfService = .userInitiated
                thread.start()
            } else {
                DispatchQueue.global(qos: .userInitiated).async(execute: work)
            }
        }
        if let hold, !raced { group.notify(queue: .global(qos: .userInitiated)) { hold.release() } }
    }

    deinit {
        cancel()
        group.wait()
        payload.deallocate()
        alternate?.deallocate()
        try? ledger.release(reservation)
    }

    package var raced: Bool { alternate != nil }
    /// Race mode: rows won by the primary (packed) and alternate (original) sources, and failed source reads.
    package func raceCounts() -> (primary: Int, alternate: Int, failures: Int) {
        lock.lock()
        defer { lock.unlock() }
        return (primaryWins, alternateWins, sourceFailures)
    }

    /// Race mode: read rows of `source` (0 primary, 1 alternate) not yet won by the other source.
    private func raceWorker(source: Int, read: (Int, UnsafeMutableRawBufferPointer) throws -> Void) {
        do {
            guard pthread_get_stacksize_np(pthread_self()) <= Self.maximumStack else {
                throw EngineError("PLE prefetch worker exceeds admitted stack bound")
            }
        } catch {
            lock.lock()
            failure = failure ?? error
            stopped = true
            finishLocked()
            lock.unlock()
            return
        }
        let base = source == 0 ? payload : alternate!
        while let slot = nextRace(source: source) {
            do {
                try read(unique[slot], UnsafeMutableRawBufferPointer(start: base.advanced(by: slot * Self.rowBytes), count: Self.rowBytes))
                lock.lock()
                if winner[slot] < 0 {
                    winner[slot] = Int8(source)
                    wonRows += 1
                    if source == 0 { primaryWins += 1 } else { alternateWins += 1 }
                    if wonRows == unique.count { finishLocked() }
                }
                lock.unlock()
            } catch {
                lock.lock()
                sourceFailures += 1
                failedSources[slot] |= UInt8(1 << source)
                if failedSources[slot] == 3 && winner[slot] < 0 {
                    failure = failure ?? error
                    stopped = true
                    finishLocked()
                }
                lock.unlock()
            }
        }
    }
    private func nextRace(source: Int) -> Int? {
        lock.lock()
        defer { lock.unlock() }
        guard !stopped, !finished else { return nil }
        if source == 0 {
            while cursor < unique.count && winner[cursor] >= 0 { cursor += 1 }
            guard cursor < unique.count else { return nil }
            defer { cursor += 1 }
            return cursor
        }
        if alternateDescending {
            while alternateCursor >= 0 && winner[alternateCursor] >= 0 { alternateCursor -= 1 }
            guard alternateCursor >= 0 else { return nil }
            defer { alternateCursor -= 1 }
            return alternateCursor
        }
        while alternateCursor < unique.count && winner[alternateCursor] >= 0 { alternateCursor += 1 }
        guard alternateCursor < unique.count else { return nil }
        defer { alternateCursor += 1 }
        return alternateCursor
    }
    /// Lock held. Completes the race once (all rows won, a row failed in both sources, or cancellation).
    private func finishLocked() {
        guard !finished else { return }
        finished = true
        complete.leave()
        if let raceHold { DispatchQueue.global(qos: .userInitiated).async { raceHold.release() } }
    }

    package var uniqueRows: Int { unique.count }

    /// True once a `wait` has joined the rows (they are then immutable for `packed` and `copyRow`).
    package var isJoined: Bool { joined }
    /// Non-blocking: every row is ready (race: every row has a winner; otherwise every reader stopped).
    /// A later `wait` then returns at once (or throws the readers' failure).
    package var readsComplete: Bool {
        if joined { return true }
        if raced {
            lock.lock()
            defer { lock.unlock() }
            return finished
        }
        return group.wait(timeout: .now()) == .success
    }

    /// `prefill_ple_overlap` host tail tiles: the joined bytes of row `index` (from the source that won it),
    /// exactly the bytes `packed` places for it. False when this scope did not read `index`.
    package func copyRow(_ index: Int, into buffer: UnsafeMutableRawBufferPointer) throws -> Bool {
        guard joined, buffer.count == Self.rowBytes else { throw EngineError("PLE prefetch row copy before its join or with a wrong size") }
        guard let slot = slots[index] else { return false }
        lock.lock()
        let fromAlternate = alternate != nil && winner[slot] == 1
        lock.unlock()
        let source = fromAlternate ? alternate! : payload
        buffer.baseAddress!.copyMemory(from: source.advanced(by: slot * Self.rowBytes), byteCount: Self.rowBytes)
        return true
    }

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
        if alternate != nil { finishLocked() }
    }
    func cancel() {
        lock.lock()
        stopped = true
        if raced { finishLocked() }
        lock.unlock()
    }

    /// Join all readers (polling the caller's cancellation) and return the seconds spent waiting; a read that does not
    /// return within the demand read bound fails with `drive_stalled` and lets the token's hold go
    /// (`ExpertReadStall.join`; the readers keep `self`).
    package func wait(shouldContinue: () -> Bool) throws -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        if !joined {
            // Race mode joins on every row having a winner; the losing source's readers finish later.
            do {
                try ExpertReadStall.join(
                    raced ? complete : group, bytes: unique.count * Self.rowBytes, what: "PLE row", shouldContinue: shouldContinue,
                    cancel: cancel)
            } catch let error as EngineError where error.code == "drive_stalled" {
                readHold?.release()
                throw error
            }
            joined = true
        }
        lock.lock()
        let error = failure
        let complete = raced ? wonRows == unique.count : cursor >= unique.count
        lock.unlock()
        if let error { throw error }
        guard complete else { throw IOOperationCancelled() }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
    }

    /// Rows of tokens `range` in hash order (16 x 100 bytes per token). Requires a completed wait.
    package func packed(tokens range: Range<Int>) throws -> [UInt8] {
        guard joined, range.lowerBound >= 0, range.upperBound <= tokenRows.count, !range.isEmpty else {
            throw EngineError("PLE prefetch packing outside its joined token range")
        }
        lock.lock()
        let winner = self.winner
        lock.unlock()
        var bytes = [UInt8](repeating: 0, count: range.count * Self.rowsPerToken * Self.rowBytes)
        bytes.withUnsafeMutableBytes { out in
            var at = 0
            for token in range {
                for index in tokenRows[token] {
                    let slot = slots[index]!
                    let source = alternate != nil && winner[slot] == 1 ? alternate! : payload
                    out.baseAddress!.advanced(by: at).copyMemory(
                        from: source.advanced(by: slot * Self.rowBytes),
                        byteCount: Self.rowBytes)
                    at += Self.rowBytes
                }
            }
        }
        return bytes
    }
}

/// `prefill_ple_overlap`: the row reader of a host-fallback (short tail) PLE tile. Rows come from the
/// joined scope prefetch, which already read every row of the scope's tokens (the same 100 bytes the
/// store returns); a row the scope did not read is read from the store and counted as a fallback
/// (must be zero: the tail's rows are a subset of the scope's).
package final class PLEScopeRowSource: @unchecked Sendable {
    private let prefetch: PLERowPrefetch
    private let store: @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void
    private let lock = NSLock()
    private var served: UInt64 = 0, missed: UInt64 = 0

    package init(prefetch: PLERowPrefetch, store: @escaping @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void) throws {
        guard prefetch.isJoined else { throw EngineError("PLE scope row source requires a joined prefetch") }
        self.prefetch = prefetch
        self.store = store
    }

    package var reader: @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void {
        { [self] index, buffer in
            if try prefetch.copyRow(index, into: buffer) {
                lock.lock()
                served += 1
                lock.unlock()
            } else {
                try store(index, buffer)
                lock.lock()
                missed += 1
                lock.unlock()
            }
        }
    }

    package func counts() -> (served: UInt64, missed: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        return (served, missed)
    }
}
