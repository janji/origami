import Darwin
import Foundation

/// One joined batch of known FIFO misses. Cache and model state remain caller-owned.
/// Host PLE path (`ReferencePLE` with known rows): replays the caller's row-cache FIFO over one tile
/// (at most 256 tokens x 16 rows) to list exactly the rows it will miss, reads them with up to eight
/// workers into one payload (100 bytes per row, miss order), and hands them back in that order. The
/// bytes are the store's rows; only their timing changes (BITWISE). The fixed admission covers the
/// workers' stack bound plus the largest payload.
package final class KnownPLERows: @unchecked Sendable {
    package static let admittedBytes: UInt64 = 66 * 1024 * 1024
    private static let maximumStack = 8 * 1024 * 1024
    private let ledger: MemoryLedger, reservation: MemoryLedger.Reservation
    private let payload: UnsafeMutableRawPointer
    package let indices: [Int]
    private let lock = NSLock()
    private var stopped = false, loaded = false
    private var failure: Error?
    private var cursor = 0

    package init(ids: [[Int]], fifo: [Int], next: Int, ledger: MemoryLedger) throws {
        guard !ids.isEmpty, ids.count <= 256, ids.allSatisfy({ $0.count == 16 }),
            ids.joined().allSatisfy({ (0..<320_001_536).contains($0) }),
            (1...4096).contains(fifo.count), fifo.indices.contains(next)
        else {
            throw EngineError("Known PLE batch exceeds admitted geometry")
        }
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: Self.admittedBytes, category: "known_ple_rows_bound")
        var resident = Set(fifo.filter { $0 >= 0 })
        var ring = fifo
        var position = next
        var missing = [Int]()
        missing.reserveCapacity(ids.count * 16)
        for index in ids.joined() where !resident.contains(index) {
            missing.append(index)
            if ring[position] >= 0 { resident.remove(ring[position]) }
            resident.insert(index)
            ring[position] = index
            position = (position + 1) % ring.count
        }
        indices = missing
        payload = .allocate(byteCount: max(1, missing.count) * 100, alignment: 16)
    }

    deinit {
        payload.deallocate()
        try? ledger.release(reservation)
    }

    private func shouldStop() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return stopped
    }
    private func fail(_ error: Error) {
        lock.lock()
        defer { lock.unlock() }
        stopped = true
        if failure == nil { failure = error }
    }

    package func load(
        shouldContinue: () -> Bool,
        read: @escaping @Sendable (Int, UnsafeMutableRawBufferPointer) throws -> Void
    ) throws {
        guard !loaded, cursor == 0, !shouldStop() else { throw EngineError("Known PLE batch cannot be reused") }
        guard shouldContinue() else { throw IOOperationCancelled() }
        let group = DispatchGroup()
        for worker in 0..<min(8, indices.count) {
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                defer { group.leave() }
                do {
                    guard pthread_get_stacksize_np(pthread_self()) <= Self.maximumStack else {
                        throw EngineError("Known PLE worker exceeds admitted stack bound")
                    }
                    for slot in stride(from: worker, to: indices.count, by: 8) {
                        if shouldStop() { break }
                        try autoreleasepool {
                            try read(indices[slot], UnsafeMutableRawBufferPointer(start: payload.advanced(by: slot * 100), count: 100))
                        }
                    }
                } catch { fail(error) }
            }
        }
        // Only the caller invokes its cancellation closure. Every writer retains
        // this charged owner until its callback and autorelease cleanup return.
        while group.wait(timeout: .now() + .milliseconds(1)) == .timedOut {
            if !shouldContinue() { fail(IOOperationCancelled()) }
        }
        if !shouldContinue() { fail(IOOperationCancelled()) }
        if let failure { throw failure }
        loaded = true
    }

    package func consume(_ index: Int) throws -> Data {
        guard loaded, cursor < indices.count, indices[cursor] == index else {
            throw EngineError("Known PLE replay differs from original FIFO misses")
        }
        defer { cursor += 1 }
        return Data(bytes: payload.advanced(by: cursor * 100), count: 100)
    }

    package func finishConsumption() throws {
        guard loaded, cursor == indices.count else { throw EngineError("Known PLE replay left unread misses") }
    }
}
