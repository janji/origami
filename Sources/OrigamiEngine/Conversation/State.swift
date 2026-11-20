import Foundation
import Metal

/// Chronological state of one GDN layer: the FP32 recurrent state [48 value heads][128][128] and the BF16
/// convolution tail (the last 3 rows of 10,240 channels), either in host arrays or published into a resident
/// Metal buffer (`GDNRecurrentStorage`). The recurrence never passes through BF16. `offset` is the token count.
public struct GDNState: Sendable {
    public package(set) var offset: Int
    private var hostRecurrent: [Float]
    package private(set) var recurrentStorage: GDNRecurrentStorage?
    /// Explicit diagnostic/legacy access materializes FP32 bits. Resident
    /// execution and checkpoint encoding use the immutable backing directly.
    public package(set) var recurrent: [Float] {
        get { recurrentStorage?.array() ?? hostRecurrent }
        set {
            hostRecurrent = newValue
            recurrentStorage = nil
        }
    }
    private var hostConvolutionTail: [UInt16]
    package private(set) var convolutionStorage: GDNRecurrentStorage?
    public package(set) var convolutionTail: [UInt16] {
        get { convolutionStorage?.withConvolutionBytes { Array($0.bindMemory(to: UInt16.self)) } ?? hostConvolutionTail }
        set {
            hostConvolutionTail = newValue
            convolutionStorage = nil
        }
    }

    public init() {
        offset = 0
        hostRecurrent = .init(repeating: 0, count: 48 * 128 * 128)
        recurrentStorage = nil
        hostConvolutionTail = .init(repeating: 0, count: 3 * 10240)
        convolutionStorage = nil
    }

    public init(offset: Int, recurrent: [Float], convolutionTail: [UInt16]) throws {
        self.offset = offset
        self.hostRecurrent = recurrent
        self.recurrentStorage = nil
        self.hostConvolutionTail = convolutionTail
        self.convolutionStorage = nil
        try validate()
    }

    public func validate() throws {
        guard (0...ModelContract.maximumContext).contains(offset), recurrentCount == 48 * 128 * 128,
            convolutionCount == 3 * 10240
        else { throw EngineError("Invalid GDN continuation state") }
    }

    var convolutionCount: Int { convolutionStorage == nil ? hostConvolutionTail.count : 3 * 10240 }
    package mutating func publishConvolution(_ storage: GDNRecurrentStorage) {
        precondition(storage.convolutionByteOffset != nil)
        convolutionStorage = storage
        hostConvolutionTail = []
    }
    func withConvolutionBytes<Result>(_ body: (UnsafeRawBufferPointer) throws -> Result) rethrows -> Result {
        if let convolutionStorage { return try convolutionStorage.withConvolutionBytes(body) }
        return try hostConvolutionTail.withUnsafeBytes(body)
    }
    var recurrentCount: Int { recurrentStorage?.count ?? hostRecurrent.count }
    package mutating func publishRecurrent(_ storage: GDNRecurrentStorage) {
        recurrentStorage = storage
        hostRecurrent = []
    }
    func withRecurrentBytes<Result>(_ body: (UnsafeRawBufferPointer) throws -> Result) rethrows -> Result {
        if let recurrentStorage { return try recurrentStorage.withBytes(body) }
        return try hostRecurrent.withUnsafeBytes(body)
    }
    /// Bounded nonthrowing copy, kept outside the rethrows callback ABI.
    /// CLT 6.3.3 can lower a Void raw-copy callback to a tail memmove
    /// without restoring x21; a rethrows caller then treats the buffer as an
    /// Error. An explicit Bool result and non-inlined boundary avoid that path.
    @inline(never)
    func copyRecurrentBytes(to destination: UnsafeMutableRawBufferPointer) -> Bool {
        let count = recurrentCount * MemoryLayout<Float>.stride
        guard destination.count == count, let pointer = destination.baseAddress else { return false }
        if let storage = recurrentStorage {
            pointer.copyMemory(from: storage.buffer.contents().advanced(by: storage.byteOffset), byteCount: count)
        } else {
            hostRecurrent.withUnsafeBytes { bytes in
                pointer.copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
            }
        }
        return true
    }
    /// Same non-inlined Bool boundary for the convolution tail (used when staging conversation frontiers).
    @inline(never)
    func copyConvolutionBytes(to destination: UnsafeMutableRawBufferPointer) -> Bool {
        guard destination.count == convolutionCount * 2, let pointer = destination.baseAddress else { return false }
        if let storage = convolutionStorage, let offset = storage.convolutionByteOffset {
            pointer.copyMemory(from: storage.buffer.contents().advanced(by: offset), byteCount: destination.count)
        } else {
            hostConvolutionTail.withUnsafeBytes { bytes in
                pointer.copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
            }
        }
        return true
    }
    package var residentCapacityBytes: UInt64 {
        (recurrentStorage?.allocationBytes ?? 0)
            + (convolutionStorage === recurrentStorage ? 0 : (convolutionStorage?.allocationBytes ?? 0))
    }
    package var hostCapacityBytes: UInt64 { UInt64(hostRecurrent.capacity * 4 + hostConvolutionTail.capacity * 2) }
    public var storageBytes: UInt64 { UInt64(recurrentCount * 4 + convolutionCount * 2) }
    public var capacityBytes: UInt64 { hostCapacityBytes + residentCapacityBytes }
}

/// BF16 QSA state. K/V are token-major [token, 2, 256], so appending does not
/// transpose a growing prefix. Pooled keys are [block,128]; raw tail has 0...3 rows.
/// Value semantics permit explicit sequence snapshots and rollback. Storage is host arrays (reference
/// sequences) or `QSAPagedStorage` pages from a shared arena (serving), whose ledger charge the arena owns.
public struct QSAState: Sendable {
    public private(set) var offset: Int = 0
    private var hostKeys: [UInt16] = []
    private var hostValues: [UInt16] = []
    private var hostPooledKeys: [UInt16] = []
    package private(set) var pages: QSAPagedStorage?
    /// Explicit reference/serialization access. The optimized path uses spans.
    public var keys: [UInt16] { get throws { try pages?.keys.array() ?? hostKeys } }
    public var values: [UInt16] { get throws { try pages?.values.array() ?? hostValues } }
    public var pooledKeys: [UInt16] { pages?.pooled.array() ?? hostPooledKeys }
    var keyCount: Int { pages?.keys.count ?? hostKeys.count }
    var pooledCount: Int { pages?.pooled.count ?? hostPooledKeys.count }
    package init(arena: QSAPageArena) { pages = QSAPagedStorage(arena: arena) }
    package init(offset: Int, pages: QSAPagedStorage, rawTail: [UInt16]) throws {
        self.offset = offset
        self.pages = pages
        self.rawTail = rawTail
        try validate()
    }
    func copyKeys(to pointer: UnsafeMutableRawPointer) throws {
        if let pages {
            try pages.keys.copy(to: pointer)
        } else {
            hostKeys.withUnsafeBytes { if !$0.isEmpty { pointer.copyMemory(from: $0.baseAddress!, byteCount: $0.count) } }
        }
    }
    func copyValues(to pointer: UnsafeMutableRawPointer) throws {
        if let pages {
            try pages.values.copy(to: pointer)
        } else {
            hostValues.withUnsafeBytes { if !$0.isEmpty { pointer.copyMemory(from: $0.baseAddress!, byteCount: $0.count) } }
        }
    }
    /// The pooled index's pages in order (64 blocks each) for the indexer's page table. A host-held index
    /// (no page arena) is copied once into a buffer from `host` and bound page by page.
    package func pooledPages(device: MTLDevice, host: ([UInt16]) throws -> MTLBuffer) throws -> [(buffer: MTLBuffer, offset: Int)] {
        if let pages { return try pages.pooled.pooledMetalBuffers(device: device).map { ($0, 0) } }
        let buffer = try host(hostPooledKeys)
        return stride(from: 0, to: hostPooledKeys.count, by: 64 * 128).map { (buffer, $0 * 2) }
    }
    public private(set) var rawTail: [UInt16] = []

    public init() {}

    public init(offset: Int, keys: [UInt16], values: [UInt16], pooledKeys: [UInt16], rawTail: [UInt16]) throws {
        self.offset = offset
        self.hostKeys = keys
        self.hostValues = values
        self.hostPooledKeys = pooledKeys
        self.rawTail = rawTail
        try validate()
    }

    public func validate() throws {
        guard (0...ModelContract.maximumContext).contains(offset), keyCount == offset * 512,
            (pages?.values.count ?? hostValues.count) == keyCount, pooledCount == (offset / 4) * 128,
            rawTail.count == (offset % 4) * 128
        else { throw EngineError("Invalid QSA continuation state") }
    }

    /// Reserve only after the caller's memory admission succeeds. Capacity is
    /// exposed separately because append growth can exceed logical content bytes.
    public mutating func reserveCapacity(tokens: Int) throws {
        guard tokens >= offset, tokens <= ModelContract.maximumContext else { throw EngineError("Invalid QSA reservation") }
        if pages == nil {
            hostKeys.reserveCapacity(tokens * 512)
            hostValues.reserveCapacity(tokens * 512)
            hostPooledKeys.reserveCapacity(tokens / 4 * 128)
        }
        rawTail.reserveCapacity(3 * 128)
    }

    /// Validates every input before publication. No old cache is modified if a
    /// layer projection, attention dispatch, or output projection fails.
    public mutating func append(
        keys newKeys: [UInt16], values newValues: [UInt16],
        pooledKeys newPooled: [UInt16], rawTail newTail: [UInt16], tokens: Int
    ) throws {
        try validate()
        guard tokens > 0, tokens <= ModelContract.maximumContext - offset,
            newKeys.count == tokens * 512, newValues.count == newKeys.count,
            newPooled.count == ((offset + tokens) / 4 - offset / 4) * 128,
            newTail.count == ((offset + tokens) % 4) * 128
        else { throw EngineError("Invalid QSA state append") }
        if var next = pages {
            // All allocations finish before publication; denial leaves every
            // old vector and all checkpoint readers unchanged.
            next.keys = try next.keys.appending(newKeys)
            next.values = try next.values.appending(newValues)
            next.pooled = try next.pooled.appending(newPooled)
            pages = next
        } else {
            hostKeys.append(contentsOf: newKeys)
            hostValues.append(contentsOf: newValues)
            hostPooledKeys.append(contentsOf: newPooled)
        }
        rawTail = newTail
        offset += tokens
    }

    package func usingPages(_ arena: QSAPageArena) throws -> QSAState {
        if pages?.arena === arena { return self }
        var result = QSAState(arena: arena)
        let k = try keys
        let v = try values
        let p = pooledKeys
        for start in stride(from: 0, to: offset, by: 256) {
            let end = min(offset, start + 256)
            try result.append(
                keys: Array(k[(start * 512)..<(end * 512)]),
                values: Array(v[(start * 512)..<(end * 512)]),
                pooledKeys: Array(p[(start / 4 * 128)..<(end / 4 * 128)]),
                rawTail: end == offset ? rawTail : [], tokens: end - start)
        }
        return result
    }

    public var storageBytes: UInt64 { UInt64((keyCount * 2 + pooledCount + rawTail.count) * 2) }
    /// Arena pages already own one shared ledger charge. Checkpoints only
    /// reserve the separate host tail and metadata, never charge pages twice.
    package var hostCapacityBytes: UInt64 {
        UInt64((hostKeys.capacity + hostValues.capacity + hostPooledKeys.capacity + rawTail.capacity) * 2)
            + (pages?.referenceCapacityBytes ?? 0)
    }
    public var capacityBytes: UInt64 {
        hostCapacityBytes + (pages.map { $0.keys.allocatedBytes + $0.values.allocatedBytes + $0.pooled.allocatedBytes } ?? 0)
    }
}
