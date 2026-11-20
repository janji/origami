import CryptoKit
import Darwin
import Foundation
import Metal

/// One prospectively admitted arena shared by the live conversation and all
/// immutable checkpoint readers. Pages keep the arena (and its charge) alive.
/// This owner never changes a published page, including its partial final page.
/// Pages are mmapped as they are written, so bytes no page uses yet are not resident: `hold` keeps only the
/// ledger charge a request can reach and lends the rest to the expert cache (see SerialExecutor.prepareUpcomingInput).
package final class QSAPageArena: @unchecked Sendable {
    package struct Snapshot: Codable, Sendable {
        package let admittedBytes, liveBytes, highWaterBytes, copiedBytes, allocations, refusals: UInt64
        /// The ledger charge held now; `admittedBytes - heldBytes` is lent to the expert cache.
        package let heldBytes: UInt64
    }
    fileprivate let ledger: MemoryLedger
    /// The charge in pieces: shrinking returns bytes in place, growing adds a reservation (the ledger never grows one).
    private var reservations: [(token: MemoryLedger.Reservation, bytes: UInt64)] = []
    package let maximumBytes: UInt64
    package let fileStore: QSAFilePageStore?
    private var fileStaging: QSAFileStaging?
    private let lock = NSLock()
    private var used: UInt64 = 0, highWater: UInt64 = 0, copies: UInt64 = 0, allocations: UInt64 = 0, refusals: UInt64 = 0
    private var held: UInt64 = 0

    package init(ledger: MemoryLedger, maximumBytes: UInt64, fileStore: QSAFilePageStore? = nil) throws {
        guard maximumBytes > 0 else { throw EngineError("Empty QSA page arena") }
        self.ledger = ledger
        self.maximumBytes = maximumBytes
        self.fileStore = fileStore
        try reserve(maximumBytes)
    }
    deinit { for piece in reservations { try? ledger.release(piece.token) } }
    private func reserve(_ bytes: UInt64) throws {
        reservations.append((try ledger.reserve(bytes: bytes, category: "conversation_qsa_page_arena"), bytes))
        held += bytes
    }
    /// Keeps `bytes` of the ledger charge (never below the live pages nor above the maximum) and returns the rest.
    /// Taking bytes back needs the ledger room: the caller shrinks the expert cache first.
    package func hold(_ bytes: UInt64) throws {
        lock.lock()
        defer { lock.unlock() }
        let target = min(maximumBytes, max(used, bytes))
        if target > held { try reserve(target - held) }
        while held > target, let last = reservations.last {
            if held - last.bytes >= target {
                try ledger.release(last.token)
                reservations.removeLast()
                held -= last.bytes
            } else {
                try ledger.shrink(last.token, to: last.bytes - (held - target))
                reservations[reservations.count - 1].bytes -= held - target
                held = target
            }
        }
    }
    package var heldBytes: UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return held
    }
    fileprivate func acquire(_ bytes: UInt64) throws {
        lock.lock()
        defer { lock.unlock() }
        guard bytes <= maximumBytes - used else {
            refusals += 1
            throw EngineError("QSA page arena exhausted: live=\(used) new=\(bytes) limit=\(maximumBytes)", code: "conversation_page_quota")
        }
        // Past what the request held (its estimate is generous, so rarely): take free ledger room, 64 MiB at a time.
        if bytes > held - used {
            do { try reserve(min(maximumBytes - held, max(bytes - (held - used), 64 << 20))) } catch {
                refusals += 1
                throw error
            }
        }
        used += bytes
        highWater = max(highWater, used)
        allocations += 1
    }
    fileprivate func release(_ bytes: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        precondition(bytes <= used)
        used -= bytes
    }
    fileprivate func copied(_ bytes: Int) {
        lock.lock()
        defer { lock.unlock() }
        copies += UInt64(bytes)
    }
    package func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            admittedBytes: maximumBytes, liveBytes: used, highWaterBytes: highWater,
            copiedBytes: copies, allocations: allocations, refusals: refusals, heldBytes: held)
    }
    package func staging(device: MTLDevice) throws -> QSAFileStaging {
        lock.lock()
        defer { lock.unlock() }
        guard let fileStore else { throw EngineError("Resident QSA arena has no file staging owner") }
        if let fileStaging {
            guard fileStaging.deviceID == device.registryID else { throw EngineError("File-KV staging belongs to another GPU") }
            return fileStaging
        }
        let value = try QSAFileStaging(device: device, ledger: ledger, store: fileStore)
        fileStaging = value
        return value
    }
    package func stagingSnapshot() -> QSAFileStaging.Snapshot? {
        lock.lock()
        defer { lock.unlock() }
        return fileStaging?.snapshot()
    }
}

private final class QSAPageAllocation: @unchecked Sendable {
    let arena: QSAPageArena, pointer: UnsafeMutableRawPointer, bytes: UInt64
    init(arena: QSAPageArena, capacity: Int) throws {
        self.arena = arena
        // mmap has exact page-rounded capacity, avoiding unbounded malloc cache
        // slack. The page is written only during construction, before publication.
        bytes = UInt64((capacity * 2 + 16_383) / 16_384 * 16_384)
        try arena.acquire(bytes)
        let result = mmap(nil, Int(bytes), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0)
        guard let result, result != MAP_FAILED else {
            arena.release(bytes)
            throw EngineError("QSA page allocation failed", code: "insufficient_memory")
        }
        pointer = result
    }
    deinit {
        munmap(pointer, Int(bytes))
        arena.release(bytes)
    }
}

private final class QSAImmutablePage: @unchecked Sendable {
    private let allocation: QSAPageAllocation
    private let lock = NSLock()
    private var buffer: MTLBuffer?
    // Immutable bytes share their digest across prompt, turn and reply snapshots.
    // The small metadata allocation remains charged for exactly this page's lifetime.
    private var checkpointDigest: (bytes: Data, memory: CheckpointMemoryLease)?
    let count: Int
    var pointer: UnsafeMutableRawPointer { allocation.pointer }
    var bytes: UInt64 { allocation.bytes }
    init(arena: QSAPageArena, capacity: Int, prefix: QSAImmutablePage?, suffix: ArraySlice<UInt16>) throws {
        let prior = prefix?.count ?? 0
        guard capacity > 0, prior + suffix.count <= capacity else { throw EngineError("QSA page geometry exceeds capacity") }
        count = prior + suffix.count
        allocation = try QSAPageAllocation(arena: arena, capacity: capacity)
        if let prefix, prior > 0 {
            pointer.copyMemory(from: prefix.pointer, byteCount: prior * 2)
            arena.copied(prior * 2)
        }
        suffix.withUnsafeBytes { if !$0.isEmpty { pointer.advanced(by: prior * 2).copyMemory(from: $0.baseAddress!, byteCount: $0.count) } }
    }
    func metalBuffer(device: MTLDevice) throws -> MTLBuffer {
        lock.lock()
        defer { lock.unlock() }
        if let buffer {
            guard buffer.device.registryID == device.registryID else { throw EngineError("Immutable QSA page belongs to a different GPU") }
            return buffer
        }
        // The buffer retains the allocation independently of the Swift page.
        // An outstanding indirect GPU reader therefore keeps both mmap storage
        // and its arena charge alive after every CPU vector reader is gone.
        let owner = allocation
        guard
            let made = device.makeBuffer(
                bytesNoCopy: pointer, length: Int(bytes), options: .storageModeShared,
                deallocator: { [owner] _, _ in withExtendedLifetime(owner) {} }), made.allocatedSize <= bytes
        else {
            throw EngineError("Shared QSA Metal page exceeds its admitted allocation", code: "insufficient_memory")
        }
        buffer = made
        return made
    }

    /// Full-size KV pages cache their root; smaller pooled pages cache the 8 KiB
    /// chunk hashes from which the unchanged on-disk 256 KiB root is assembled.
    func digest() -> Data {
        lock.lock()
        defer { lock.unlock() }
        if let checkpointDigest { return checkpointDigest.bytes }
        let bytes = UnsafeRawBufferPointer(start: pointer, count: count * 2)
        let result: Data
        if allocation.bytes >= UInt64(PagedCheckpointFormat.pageBytes) {
            result = PagedCheckpointFormat.root(bytes)
        } else {
            var chunks = Data()
            for start in stride(from: 0, to: bytes.count, by: QSAFilePageStore.chunkBytes) {
                var hash = SHA256()
                hash.update(
                    bufferPointer: UnsafeRawBufferPointer(rebasing: bytes[start..<min(bytes.count, start + QSAFilePageStore.chunkBytes)]))
                hash.finalize().withUnsafeBytes { chunks.append(contentsOf: $0) }
            }
            result = chunks
        }
        // Cache admission is optional: a full owner budget can still save safely.
        if let memory = try? CheckpointMemoryLease(
            ledger: allocation.arena.ledger,
            bytes: UInt64(result.count + 256), category: "checkpoint_page_digests")
        {
            checkpointDigest = (result, memory)
        }
        return result
    }
}

/// Persistent vector; an append copies at most its last partial page. Copying
/// this value shares immutable pages, never a growing contiguous tensor.
package struct QSAPagedVector: Sendable {
    private var pages: [QSAImmutablePage] = []
    package private(set) var count = 0
    private let arena: QSAPageArena
    private let pageElements: Int
    package init(arena: QSAPageArena, pageElements: Int) {
        self.arena = arena
        self.pageElements = pageElements
    }
    package func appending(_ values: [UInt16]) throws -> Self {
        guard !values.isEmpty else { return self }
        var result = self
        var cursor = 0
        if let tail = pages.last, tail.count < pageElements {
            let amount = min(pageElements - tail.count, values.count)
            let replacement = try QSAImmutablePage(arena: arena, capacity: pageElements, prefix: tail, suffix: values[0..<amount])
            result.pages[result.pages.count - 1] = replacement
            cursor = amount
        }
        while cursor < values.count {
            let end = min(values.count, cursor + pageElements)
            result.pages.append(try QSAImmutablePage(arena: arena, capacity: pageElements, prefix: nil, suffix: values[cursor..<end]))
            cursor = end
        }
        result.count += values.count
        return result
    }
    package func withSpans(_ body: (Int, UnsafeRawBufferPointer) throws -> Void) rethrows {
        var offset = 0
        for page in pages {
            try body(offset, UnsafeRawBufferPointer(start: page.pointer, count: page.count * 2))
            offset += page.count
        }
    }
    /// Metadata-only traversal for unchanged pages. Payload is copied only when
    /// the writer asks for a changed page, preserving the existing disk format.
    /// Each loader owns its immutable page group, so forwarding it through the
    /// checkpoint callback does not borrow another callback's exclusive access.
    package func withCheckpointPages(
        _ body: (
            Int, Int, Data,
            @escaping (UnsafeMutableRawBufferPointer) throws -> Void
        ) throws -> Void
    ) throws {
        let pageBytes = PagedCheckpointFormat.pageBytes
        guard
            pageElements * 2 == pageBytes
                || (pageElements * 2 < pageBytes && pageBytes % (pageElements * 2) == 0
                    && pageElements * 2 % QSAFilePageStore.chunkBytes == 0)
        else {
            throw EngineError("Resident checkpoint page geometry differs")
        }
        let perPage = pageBytes / (pageElements * 2)
        for first in stride(from: 0, to: pages.count, by: perPage) {
            let group = pages[first..<min(pages.count, first + perPage)]
            let count = group.reduce(0) { $0 + $1.count * 2 }
            let root: Data
            if perPage == 1 {
                root = group.first!.digest()
            } else {
                var hash = SHA256()
                var length = UInt64(count).littleEndian
                withUnsafeBytes(of: &length) { hash.update(bufferPointer: $0) }
                for page in group { page.digest().withUnsafeBytes { hash.update(bufferPointer: $0) } }
                root = Data(hash.finalize())
            }
            try body(first / perPage, count, root) { target in
                guard target.count == count else { throw EngineError("Resident checkpoint page size differs") }
                var offset = 0
                for page in group {
                    let bytes = page.count * 2
                    target.baseAddress!.advanced(by: offset).copyMemory(from: page.pointer, byteCount: bytes)
                    offset += bytes
                }
            }
        }
    }
    package func copy(to target: UnsafeMutableRawPointer) {
        withSpans { offset, bytes in target.advanced(by: offset * 2).copyMemory(from: bytes.baseAddress!, byteCount: bytes.count) }
    }
    package func array() -> [UInt16] {
        [UInt16](unsafeUninitializedCapacity: count) { buffer, initialized in
            if let pointer = buffer.baseAddress { copy(to: UnsafeMutableRawPointer(pointer)) }
            initialized = count
        }
    }
    package var storageBytes: UInt64 { UInt64(count * 2) }
    package var referenceCapacityBytes: UInt64 { UInt64(pages.capacity * MemoryLayout<QSAImmutablePage>.stride) }
    package var allocatedBytes: UInt64 { pages.reduce(0) { $0 + $1.bytes } }
    package var pageIdentities: Set<ObjectIdentifier> { Set(pages.map(ObjectIdentifier.init)) }
    package func metalBuffers(device: MTLDevice) throws -> [MTLBuffer] {
        guard pageElements == 256 * 512 else { throw EngineError("QSA Metal KV pages have a different token geometry") }
        return try pages.map { try $0.metalBuffer(device: device) }
    }
    /// The pooled index's pages as GPU buffers (64 blocks of 128 values each), for the indexer's page table.
    package func pooledMetalBuffers(device: MTLDevice) throws -> [MTLBuffer] {
        guard pageElements == 64 * 128 else { throw EngineError("QSA pooled pages have a different block geometry") }
        return try pages.map { try $0.metalBuffer(device: device) }
    }
}

/// The same persistent KV contract with an explicit storage representation.
/// File reads throw; a failed read can never appear as an empty/zero tensor.
package enum QSAKeyPages: Sendable {
    case memory(QSAPagedVector)
    case file(QSAFileVector)
    package init(arena: QSAPageArena) {
        if let store = arena.fileStore {
            self = .file(QSAFileVector(store: store))
        } else {
            self = .memory(QSAPagedVector(arena: arena, pageElements: 256 * 512))
        }
    }
    package var count: Int {
        switch self {
        case .memory(let v): return v.count
        case .file(let v): return v.count
        }
    }
    package var referenceCapacityBytes: UInt64 {
        switch self {
        case .memory(let v): return v.referenceCapacityBytes
        case .file(let v): return v.referenceCapacityBytes
        }
    }
    package var allocatedBytes: UInt64 {
        switch self {
        case .memory(let v): return v.allocatedBytes
        case .file: return 0
        }
    }
    package var pageIdentities: Set<ObjectIdentifier> {
        switch self {
        case .memory(let v): return v.pageIdentities
        case .file(let v): return v.pageIdentities
        }
    }
    package func appending(_ values: [UInt16]) throws -> Self {
        switch self {
        case .memory(let v): return .memory(try v.appending(values))
        case .file(let v): return .file(try v.appending(values))
        }
    }
    package func array() throws -> [UInt16] {
        switch self {
        case .memory(let v): return v.array()
        case .file(let v): return try v.array()
        }
    }
    package func copy(to pointer: UnsafeMutableRawPointer) throws {
        switch self {
        case .memory(let v): v.copy(to: pointer)
        case .file(let v): try v.copy(to: pointer)
        }
    }
    package func metalBuffers(device: MTLDevice) throws -> [MTLBuffer] {
        guard case .memory(let v) = self else { throw EngineError("File KV requires a verified staging lease") }
        return try v.metalBuffers(device: device)
    }
}

/// One QSA layer's persistent state: keys and values (256-token pages of 512 BF16 per token) and the
/// pooled index (64 blocks x 128 BF16 per page), all charged to one arena.
package struct QSAPagedStorage: Sendable {
    // All 36 vectors at 1024 pages, geometric Array growth, allocator slack and
    // headers. Payload mappings/page objects retain their independent owners.
    package static let maximumReferenceBytes: UInt64 = 2 * 1024 * 1024
    package let arena: QSAPageArena
    package var keys, values: QSAKeyPages
    package var pooled: QSAPagedVector
    package var referenceCapacityBytes: UInt64 {
        keys.referenceCapacityBytes + values.referenceCapacityBytes + pooled.referenceCapacityBytes
    }
    package init(arena: QSAPageArena) {
        self.arena = arena
        keys = QSAKeyPages(arena: arena)
        values = QSAKeyPages(arena: arena)
        pooled = QSAPagedVector(arena: arena, pageElements: 64 * 128)
    }
}
