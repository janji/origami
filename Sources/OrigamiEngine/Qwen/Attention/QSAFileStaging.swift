import Darwin
import Foundation
import Metal
import ObjectiveC

private final class QSAFileFrameAllocation: @unchecked Sendable {
    let pointer: UnsafeMutableRawPointer
    private let memory: CheckpointMemoryLease
    init(ledger: MemoryLedger) throws {
        memory = try CheckpointMemoryLease(
            ledger: ledger, bytes: QSAFileStaging.admittedBytes, category: "conversation_file_kv_staging_cache")
        guard let p = mmap(nil, QSAFileStaging.mappedBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0), p != MAP_FAILED else {
            throw EngineError("File-KV frame/cache allocation failed", code: "insufficient_memory")
        }
        pointer = p
    }
    deinit { munmap(pointer, QSAFileStaging.mappedBytes) }
}
private enum QSAFileLeaseAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// One shared full-layer frame and a bounded cache of verified immutable chunks.
/// Frame positions retain absolute token addresses, preserving attention's
/// original lane/partition arithmetic. A sparse lease authorizes initialized
/// chunks only. No second GPU reader may overlap a frame mutation.
/// One 1 GiB anonymous mapping: a 512 MiB frame (keys in the first half, values in the second, page n
/// at n x 256 KiB, so 1024 pages = 262,144 tokens per half) wrapped as one Metal buffer, then a 480 MiB
/// LRU cache of verified 8 KiB chunks keyed by (slot, chunk, generation). Prefill stages whole layers;
/// decode stages only the chunks its <= 2051 selected keys touch.
package final class QSAFileStaging: @unchecked Sendable {
    package static let admittedBytes: UInt64 = 1 << 30
    package static let frameBytes = 512 * 1024 * 1024
    private static let cacheBytes = 480 * 1024 * 1024
    fileprivate static let mappedBytes = frameBytes + cacheBytes
    private static let cacheCapacity = cacheBytes / QSAFilePageStore.chunkBytes
    /// A chunk of one slot generation and its length (views of a slot share its full chunks, not a partial one):
    /// (slot × 32 + chunk) << 14 | bytes, so the key stays two words and the index within its escrow.
    private struct Key: Hashable { let slotChunk, generation: UInt64 }
    private struct Entry {
        var key: Key?
        var previous = -1, next = -1, bytes = 0
    }
    package struct Snapshot: Codable, Sendable {
        package let admittedBytes, fileReadBytes, cacheHitBytes: UInt64
        package let cacheEntries, transactions: Int
        package let active: Bool
    }
    /// The right of one command to read the frame. The command and its completion handler retain it, so
    /// the frame cannot be restaged or freed while a submitted reader runs; `finish` releases it after join.
    package final class Lease: @unchecked Sendable {
        package let owner: QSAFileStaging, buffer: MTLBuffer
        package let oldTokens: Int, sparse: Bool
        fileprivate let id: UUID
        fileprivate weak var command: MTLCommandBuffer?
        fileprivate var finished = false
        fileprivate init(owner: QSAFileStaging, command: MTLCommandBuffer, oldTokens: Int, sparse: Bool, id: UUID) {
            self.owner = owner
            self.command = command
            self.oldTokens = oldTokens
            self.sparse = sparse
            self.id = id
            buffer = owner.frame
        }
        deinit { if !finished { owner.abandon(id) } }
        package func validateSelection(_ selected: [[UInt32]], totalTokens: Int) throws {
            try owner.validateSelection(self, selected: selected, totalTokens: totalTokens)
        }
        package func finish(command: MTLCommandBuffer, success: Bool) throws {
            try owner.finish(self, command: command, success: success)
        }
    }
    private let allocation: QSAFileFrameAllocation, frame: MTLBuffer, store: QSAFilePageStore
    private let cache: UnsafeMutableRawPointer
    private var entries: [Entry], index: [Key: Int] = [:]
    private var used = 0, newest = -1, oldest = -1
    private var keys = [QSAFilePageStore.Page?](repeating: nil, count: 1024)
    private var values = [QSAFilePageStore.Page?](repeating: nil, count: 1024)
    private var keyMask = [UInt32](repeating: 0, count: 1024), valueMask = [UInt32](repeating: 0, count: 1024)
    private let lock = NSLock()
    private var active: UUID?
    private var fileReadBytes: UInt64 = 0, cacheHitBytes: UInt64 = 0
    private var transactions = 0
    package var deviceID: UInt64 { frame.device.registryID }

    package init(device: MTLDevice, ledger: MemoryLedger, store: QSAFilePageStore) throws {
        let allocation = try QSAFileFrameAllocation(ledger: ledger)
        guard
            let frame = device.makeBuffer(
                bytesNoCopy: allocation.pointer, length: Self.frameBytes, options: .storageModeShared,
                deallocator: { [allocation] _, _ in withExtendedLifetime(allocation) {} }), frame.allocatedSize <= Self.frameBytes
        else {
            throw EngineError("File-KV Metal frame exceeds admission", code: "insufficient_memory")
        }
        self.allocation = allocation
        self.frame = frame
        self.store = store
        cache = allocation.pointer.advanced(by: Self.frameBytes)
        entries = [Entry](repeating: Entry(), count: Self.cacheCapacity)
        index.reserveCapacity(Self.cacheCapacity)
        // Conservative Swift Array/Dictionary allocator slack. Fixed cardinality
        // prevents rehash growth; all frame metadata is inside the 32 MiB escrow.
        let metadata =
            2 * entries.capacity * MemoryLayout<Entry>.stride
            + 4 * index.capacity * (MemoryLayout<Key>.stride + MemoryLayout<Int>.stride + 32) + 1_048_576
        guard metadata <= 32 * 1024 * 1024, index.capacity >= Self.cacheCapacity else {
            throw EngineError("File-KV cache metadata exceeds its fixed escrow", code: "insufficient_memory")
        }
    }
    private func touch(_ slot: Int) {
        guard newest != slot else { return }
        let p = entries[slot].previous
        let n = entries[slot].next
        if p >= 0 { entries[p].next = n }
        if n >= 0 { entries[n].previous = p }
        if oldest == slot { oldest = p }
        entries[slot].previous = -1
        entries[slot].next = newest
        if newest >= 0 { entries[newest].previous = slot }
        newest = slot
        if oldest < 0 { oldest = slot }
    }
    private func cached(_ page: QSAFilePageStore.Page, chunk: Int, target: UnsafeMutableRawPointer) throws {
        let identity = page.cacheIdentity
        let count = min(QSAFilePageStore.chunkBytes, page.count * 2 - chunk * QSAFilePageStore.chunkBytes)
        let key = Key(slotChunk: UInt64(identity.slot * 32 + chunk) << 14 | UInt64(count), generation: identity.generation)
        guard count > 0 else { throw EngineError("File-KV cache requested an empty unit") }
        if let slot = index[key] {
            guard entries[slot].key == key, entries[slot].bytes == count else { throw EngineError("File-KV cache identity differs") }
            target.copyMemory(from: cache.advanced(by: slot * QSAFilePageStore.chunkBytes), byteCount: count)
            cacheHitBytes += UInt64(count)
            touch(slot)
            return
        }
        let begin = chunk * QSAFilePageStore.chunkBytes
        try store.read(page, range: begin..<(begin + count), into: UnsafeMutableRawBufferPointer(start: target, count: count))
        fileReadBytes += UInt64(count)
        let slot: Int
        if used < Self.cacheCapacity {
            slot = used
            used += 1
        } else {
            slot = oldest
            guard slot >= 0, let old = entries[slot].key, index.removeValue(forKey: old) == slot else {
                throw EngineError("File-KV cache eviction chain differs")
            }
        }
        cache.advanced(by: slot * QSAFilePageStore.chunkBytes).copyMemory(from: target, byteCount: count)
        entries[slot].key = key
        entries[slot].bytes = count
        index[key] = slot
        touch(slot)
        guard index.count == used, used <= Self.cacheCapacity else { throw EngineError("File-KV cache exceeds fixed cardinality") }
    }
    private func stageVector(_ vector: QSAFileVector, value: Bool, chunks: [UInt32]?) throws {
        guard vector.store === store, vector.count <= ModelContract.maximumContext * 512 else {
            throw EngineError("File-KV staging owner/extent differs")
        }
        let base = value ? Self.frameBytes / 2 : 0
        for (number, page) in vector.pages.enumerated() {
            let previous = value ? values[number] : keys[number]
            if previous !== page {
                if value {
                    values[number] = page
                    valueMask[number] = 0
                } else {
                    keys[number] = page
                    keyMask[number] = 0
                }
            }
            let pointer = frame.contents().advanced(by: base + number * QSAFilePageStore.pageBytes)
            let count = (page.count * 2 + QSAFilePageStore.chunkBytes - 1) / QSAFilePageStore.chunkBytes
            let full: UInt32 = count == 32 ? .max : (UInt32(1) << count) - 1
            var mask = value ? valueMask[number] : keyMask[number]
            if chunks == nil {
                if mask != full {
                    // Clear authorization before any possibly failing overwrite.
                    if value { valueMask[number] = 0 } else { keyMask[number] = 0 }
                    try store.read(
                        page, range: 0..<(page.count * 2), into: UnsafeMutableRawBufferPointer(start: pointer, count: page.count * 2))
                    fileReadBytes += UInt64(page.count * 2)
                    mask = full
                }
            } else {
                var missing = chunks![number] & ~mask
                while missing != 0 {
                    let chunk = missing.trailingZeroBitCount
                    guard chunk < count else { throw EngineError("File-KV staging selected a missing chunk") }
                    try cached(page, chunk: chunk, target: pointer.advanced(by: chunk * QSAFilePageStore.chunkBytes))
                    mask |= UInt32(1) << chunk
                    missing &= missing - 1
                }
            }
            if value { valueMask[number] = mask } else { keyMask[number] = mask }
        }
        // Release old-layer/old-branch file readers beyond the new frontier.
        for number in vector.pages.count..<1024 {
            if value {
                values[number] = nil
                valueMask[number] = 0
            } else {
                keys[number] = nil
                keyMask[number] = 0
            }
        }
    }
    package func stage(
        command: MTLCommandBuffer, keys: QSAFileVector, values: QSAFileVector,
        rows: Int, selected: [[UInt32]]? = nil
    ) throws -> Lease {
        guard lock.try() else { throw EngineError("File-KV staging owner is busy") }
        defer { lock.unlock() }
        guard active == nil, command.status == .notEnqueued, command.commandQueue.device.registryID == frame.device.registryID,
            keys.count == values.count, keys.count % 512 == 0, (1...256).contains(rows),
            keys.count / 512 <= ModelContract.maximumContext - rows
        else {
            throw EngineError("File-KV staging frontier/command differs")
        }
        let old = keys.count / 512
        var chunks: [UInt32]?
        if let selected {
            guard rows <= 2, selected.count == rows else { throw EngineError("Sparse file-KV staging requires vector rows") }
            var needed = [UInt32](repeating: 0, count: 1024)
            for (row, positions) in selected.enumerated() {
                guard positions.count <= PinnedAttention.maximumSelectedKeys, zip(positions, positions.dropFirst()).allSatisfy({ $0 < $1 }),
                    positions.allSatisfy({ Int($0) <= old + row })
                else { throw EngineError("Sparse file-KV selection is not bounded/causal") }
                for token in positions where Int(token) < old { needed[Int(token) / 256] |= UInt32(1) << (Int(token) % 256 / 8) }
            }
            chunks = needed
        }
        try stageVector(keys, value: false, chunks: chunks)
        try stageVector(values, value: true, chunks: chunks)
        let id = UUID()
        let lease = Lease(owner: self, command: command, oldTokens: old, sparse: selected != nil, id: id)
        active = id
        transactions += 1
        // The command and completion handler independently retain the lease.
        // Its weak command link avoids a cycle; a discarded command cannot
        // release the frame while a submitted GPU reader is still executing.
        objc_setAssociatedObject(command, &QSAFileLeaseAssociation.key, lease, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        command.addCompletedHandler { [lease] _ in withExtendedLifetime(lease) {} }
        return lease
    }
    private func validateSelection(_ lease: Lease, selected: [[UInt32]], totalTokens: Int) throws {
        lock.lock()
        defer { lock.unlock() }
        guard !lease.finished, active == lease.id, totalTokens == lease.oldTokens + selected.count,
            (1...2).contains(selected.count), selected.allSatisfy({ $0.count <= PinnedAttention.maximumSelectedKeys })
        else {
            throw EngineError("File-KV selection does not belong to the active vector lease")
        }
        for (row, positions) in selected.enumerated() {
            guard zip(positions, positions.dropFirst()).allSatisfy({ $0 < $1 }) else {
                throw EngineError("File-KV selection order differs")
            }
            for token in positions {
                guard Int(token) <= lease.oldTokens + row else { throw EngineError("File-KV selection is outside its causal frontier") }
                if Int(token) < lease.oldTokens {
                    let page = Int(token) / 256
                    let bit = UInt32(1) << (Int(token) % 256 / 8)
                    guard keyMask[page] & bit != 0, valueMask[page] & bit != 0 else {
                        throw EngineError("File-KV attention requested an uninitialized span")
                    }
                }
            }
        }
    }
    private func finish(_ lease: Lease, command: MTLCommandBuffer, success: Bool) throws {
        lock.lock()
        defer { lock.unlock() }
        guard !lease.finished, active == lease.id, lease.command === command,
            success ? command.status == .completed : [.notEnqueued, .completed, .error].contains(command.status)
        else {
            throw EngineError("File-KV frame reader has not joined or belongs to another lease")
        }
        active = nil
        lease.finished = true
    }
    private func abandon(_ id: UUID) {
        lock.lock()
        defer { lock.unlock() }
        if active == id { active = nil }
    }
    package func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            admittedBytes: Self.admittedBytes, fileReadBytes: fileReadBytes, cacheHitBytes: cacheHitBytes,
            cacheEntries: used, transactions: transactions, active: active != nil)
    }
}
