import CExactIO
import Darwin
import Foundation
import Metal
import ObjectiveC

private final class ResidentMetadataOwner {
    let ledger: MemoryLedger
    let reservation: MemoryLedger.Reservation
    init(ledger: MemoryLedger, bytes: UInt64) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: "resident-tensor-metadata")
    }
    deinit { try? ledger.release(reservation) }
}

/// Independent of the cache lock: raw resources can retire on a Metal thread.
private final class ResidentAllocationBudget {
    private let lock = NSLock()
    private var bytes: UInt64 = 0
    let limit: UInt64
    init(limit: UInt64) { self.limit = limit }
    var allocated: UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return bytes
    }
    func reserve(_ count: UInt64) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard count <= limit - bytes else { return false }
        bytes += count
        return true
    }
    func release(_ count: UInt64) {
        lock.lock()
        bytes -= count
        lock.unlock()
    }
}

private final class ResidentAllocationOwner {
    let metadata: ResidentMetadataOwner
    let budget: ResidentAllocationBudget
    let reservation: MemoryLedger.Reservation
    let bytes: UInt64
    init(
        metadata: ResidentMetadataOwner, budget: ResidentAllocationBudget,
        reservation: MemoryLedger.Reservation, bytes: UInt64
    ) {
        self.metadata = metadata
        self.budget = budget
        self.reservation = reservation
        self.bytes = bytes
    }
    deinit {
        try? metadata.ledger.release(reservation)
        budget.release(bytes)
    }
}
private enum ResidentAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// `wire_decode_slabs` also wires (mlock) the resident dense weights: every decode token reads them, and under memory
/// pressure the VM compresses pages only the GPU or BLAS reads, so each command then waits on decompression. Returns
/// the bytes wired, or nil when the wire failed (the buffer stays usable, unwired).
func wireResidentPages(_ buffer: MTLBuffer) -> Int? {
    let page = Int(getpagesize())
    let length = buffer.allocatedSize
    // shortcut: a sub-page buffer Metal packs beside others (a norm vector) stays unwired, since wiring its page would
    // pin and later unpin its neighbours; give small tensors their own pages if they ever matter.
    guard length % page == 0, Int(bitPattern: buffer.contents()) % page == 0 else { return 0 }
    var pages: UInt64 = 0
    return origami_wire_pages(buffer.contents(), length, page, &pages) == 0 ? length : nil
}

private final class ResidentTensorEntry {
    let name: String
    let span: TensorSpan
    let excludedByDefault: Bool
    let buffer: MTLBuffer
    let chargedBytes: UInt64
    /// Pages wired at admission; unwired when the last entry owner drops, while the buffer is still alive.
    let wiredBytes: Int
    var stamp: UInt64
    private let lock = NSLock()
    private var pins = 0
    init(name: String, span: TensorSpan, excludedByDefault: Bool, buffer: MTLBuffer, chargedBytes: UInt64, wiredBytes: Int, stamp: UInt64) {
        self.name = name
        self.span = span
        self.buffer = buffer
        self.excludedByDefault = excludedByDefault
        self.chargedBytes = chargedBytes
        self.wiredBytes = wiredBytes
        self.stamp = stamp
    }
    deinit { if wiredBytes > 0 { _ = origami_unwire_pages(buffer.contents(), wiredBytes) } }
    var pinned: Bool {
        lock.lock()
        defer { lock.unlock() }
        return pins != 0
    }
    func pin() {
        lock.lock()
        pins += 1
        lock.unlock()
    }
    func unpin() {
        lock.lock()
        pins -= 1
        lock.unlock()
    }
}

/// Read-only by contract. Never write through buffer.contents() or a GPU encoder.
/// Hold this lease through GPU completion to protect LRU residency. The buffer
/// is never reused; even a raw buffer escaping the lease retains its allocation
/// charge until Metal and all other owners release the physical resource.
public final class ResidentTensorLease: @unchecked Sendable {
    private let entry: ResidentTensorEntry
    public var name: String { entry.name }
    public var span: TensorSpan { entry.span }
    public var buffer: MTLBuffer { entry.buffer }
    public var length: Int { Int(entry.span.length) }
    fileprivate init(_ entry: ResidentTensorEntry) {
        self.entry = entry
        entry.pin()
    }
    deinit { entry.unpin() }
}

/// Optional exact-byte fixed-weight residency. Synchronous reads and LRU
/// management serialize under one lock; no extra workers, copies or predictors.
/// A cache belongs to one validated PackStore/source-manifest identity.
public final class ResidentTensorCache: @unchecked Sendable {
    /// Budget, occupancy and hit/miss/eviction/bypass counters; bytes the cached entries hold wired and failed wires.
    public struct Snapshot: Codable, Sendable {
        public let byteLimit, allocatedBytes, cachedBytes, cachedAllocationBytes: UInt64
        public let entries, pinnedEntries: Int
        public let hits, misses, evictions, readBytes, bypasses: UInt64
        public let wiredBytes, wireFallbacks: UInt64
    }
    public let sourceManifestSHA256: String
    public let maximumTensorBytes: UInt64
    private let store: PackStore
    private let device: MTLDevice
    private let metadata: ResidentMetadataOwner
    private let budget: ResidentAllocationBudget
    private let lock = NSLock()
    private let maximumEntries: Int
    private let wire: Bool
    private let pageBytes = UInt64(getpagesize())
    private var entries: [String: ResidentTensorEntry] = [:]
    private var stamp: UInt64 = 0
    private var cachedBytes: UInt64 = 0, cachedAllocationBytes: UInt64 = 0
    private var hits: UInt64 = 0, misses: UInt64 = 0, evictions: UInt64 = 0
    private var readBytes: UInt64 = 0, bypasses: UInt64 = 0
    private var wiredBytes: UInt64 = 0, wireFallbacks: UInt64 = 0

    /// `wire` (from `wire_decode_slabs`) wires each new entry after its bytes are read; the ledger charge is the same.
    public init(
        store: PackStore, device: MTLDevice, ledger: MemoryLedger,
        byteLimit: UInt64, maximumTensorBytes: UInt64 = 400_000_000,
        maximumEntries: Int = 4096, wire: Bool = false
    ) throws {
        guard (1...400_000_000).contains(maximumTensorBytes), (1...4096).contains(maximumEntries) else {
            throw EngineError("Invalid resident tensor size or entry bound")
        }
        self.store = store
        self.device = device
        self.maximumTensorBytes = maximumTensorBytes
        self.maximumEntries = maximumEntries
        self.wire = wire
        sourceManifestSHA256 = store.modelFingerprint
        budget = ResidentAllocationBudget(limit: byteLimit)
        // Includes bounded dictionary/entry/lease bookkeeping. This owner also
        // follows escaped buffers so cache destruction cannot retire it early.
        metadata = try ResidentMetadataOwner(ledger: ledger, bytes: UInt64(maximumEntries * 512 + 65_536))
    }

    private func canonical(_ name: String) -> String {
        name.hasPrefix("language_model.") ? name : "language_model." + name
    }
    private func excluded(_ name: String) -> Bool {
        name.contains(".switch_mlp.") || name.contains(".ple_embedding.ngram_embedding.")
            || name.hasPrefix("language_model.model.embed_tokens.")
    }
    private func geometry(_ name: String) throws -> TensorSpan {
        guard let span = store.tensors[name] else { throw EngineError("Missing resident tensor: \(name)") }
        let widths = [
            "BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "F16": 2,
            "BF16": 2, "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8,
        ]
        guard let width = widths[span.dtype], span.offset >= 0, span.length > 0,
            !span.shape.isEmpty, span.shape.allSatisfy({ $0 > 0 })
        else {
            throw EngineError("Invalid resident tensor dtype or span: \(name)")
        }
        var bytes = width
        for dimension in span.shape {
            let product = bytes.multipliedReportingOverflow(by: dimension)
            guard !product.overflow else { throw EngineError("Resident tensor shape overflow: \(name)") }
            bytes = product.partialValue
        }
        guard Int64(bytes) == span.length else { throw EngineError("Resident tensor shape/byte mismatch: \(name)") }
        return span
    }
    private func charge(_ span: TensorSpan) -> UInt64 {
        // Validated positive Int64 lengths cannot overflow this UInt64 rounding.
        (UInt64(span.length) + pageBytes - 1) / pageBytes * pageBytes
    }
    private func touch(_ entry: ResidentTensorEntry) {
        // Saturation remains deterministic; wrapping would make new data oldest.
        stamp = stamp == UInt64.max ? stamp : stamp + 1
        entry.stamp = stamp
    }
    private func evict(_ name: String) {
        autoreleasepool {
            guard let entry = entries.removeValue(forKey: name) else { return }
            cachedBytes -= UInt64(entry.span.length)
            cachedAllocationBytes -= entry.chargedBytes
            wiredBytes -= UInt64(entry.wiredBytes)
            evictions &+= 1
        }
    }

    public func acquire(_ name: String, allowExcluded: Bool = false) throws -> ResidentTensorLease? {
        try acquire([name], allowExcluded: allowExcluded)?.first
    }

    /// Preserves requested order (including duplicates), with group admission:
    /// weights, scales and biases cannot evict each other. Nil means an optional
    /// cache bypass, never a corrupt-source/read failure; those errors propagate.
    /// Excluded expert/PLE/embedding matrices require explicit opt-in and still
    /// obey the 400 MB individual bound. No tensor rows are cached here.
    public func acquire(_ names: [String], allowExcluded: Bool = false) throws -> [ResidentTensorLease]? {
        guard !names.isEmpty, names.count <= 16 else { throw EngineError("Resident acquisition group must contain 1...16 tensors") }
        lock.lock()
        defer { lock.unlock() }
        let ordered = names.map(canonical)
        let unique = Array(Set(ordered)).sorted()
        let wanted = Set(unique)
        // Check identity even for all-hit requests. PackStore validates each
        // original descriptor; a changed source cannot silently reuse old bytes.
        try store.validateBatchBoundary()
        var spans: [String: TensorSpan] = [:]
        var missingBytes: UInt64 = 0
        var wantedBytes: UInt64 = 0
        var missingCount = 0
        for name in unique {
            // The manifest and admission policy are immutable for this cache.
            // Source mutation is still checked above, including all-hit groups.
            let entry = entries[name]
            let span = try entry?.span ?? geometry(name)
            guard allowExcluded || !(entry?.excludedByDefault ?? excluded(name)), UInt64(span.length) <= maximumTensorBytes else {
                bypasses &+= 1
                return nil
            }
            spans[name] = span
            wantedBytes += charge(span)
            if entry == nil {
                missingBytes += charge(span)
                missingCount += 1
            }
        }
        guard wantedBytes <= budget.limit, unique.count <= maximumEntries else {
            bypasses &+= 1
            return nil
        }
        // Victim list holds only names/stamps, never buffer references that would
        // delay releasing physical memory until after admission.
        if missingBytes > budget.limit - budget.allocated || entries.count + missingCount > maximumEntries {
            var victims: [(String, UInt64)] = []
            for (name, entry) in entries where !wanted.contains(name) && !entry.pinned {
                victims.append((name, entry.stamp))
            }
            victims.sort { lhs, rhs in lhs.1 == rhs.1 ? lhs.0 < rhs.0 : lhs.1 < rhs.1 }
            for (name, _) in victims {
                if missingBytes <= budget.limit - budget.allocated && entries.count + missingCount <= maximumEntries { break }
                evict(name)
            }
        }
        guard missingBytes <= budget.limit - budget.allocated, entries.count + missingCount <= maximumEntries else {
            bypasses &+= 1
            return nil
        }
        // One local autorelease pool per allocation/read makes LRU reuse work
        // inside long-running server requests without waiting for an outer pool.
        for name in unique {
            if let entry = entries[name] {
                touch(entry)
                hits &+= 1
                continue
            }
            try admit(name, span: spans[name]!)
        }
        return ordered.map { ResidentTensorLease(entries[$0]!) }
    }

    /// Caller holds the lock and has checked capacity. One miss: charge, allocate, read, insert.
    private func admit(_ name: String, span: TensorSpan) throws {
        let bytes = charge(span)
        misses &+= 1
        let entry = try autoreleasepool { () throws -> ResidentTensorEntry in
            guard budget.reserve(bytes) else { throw EngineError("Resident admission changed while locked") }
            let reservation: MemoryLedger.Reservation
            do { reservation = try metadata.ledger.reserve(bytes: bytes, category: "resident-tensors") } catch {
                budget.release(bytes)
                throw error
            }
            let owner = ResidentAllocationOwner(metadata: metadata, budget: budget, reservation: reservation, bytes: bytes)
            guard let buffer = device.makeBuffer(length: Int(span.length), options: .storageModeShared) else {
                throw EngineError("Resident tensor Metal allocation failed", code: "insufficient_memory")
            }
            objc_setAssociatedObject(buffer, &ResidentAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
            guard UInt64(buffer.allocatedSize) <= bytes else {
                throw EngineError("Metal tensor allocation exceeded its reserved page bound")
            }
            try store.readTensor(name, into: UnsafeMutableRawBufferPointer(start: buffer.contents(), count: Int(span.length)))
            readBytes &+= UInt64(span.length)
            let wired = wire ? wireResidentPages(buffer) : 0
            if let wired { wiredBytes += UInt64(wired) } else { wireFallbacks &+= 1 }
            let entry = ResidentTensorEntry(
                name: name, span: span, excludedByDefault: excluded(name),
                buffer: buffer, chargedBytes: bytes, wiredBytes: wired ?? 0, stamp: 0)
            touch(entry)
            return entry
        }
        entries[name] = entry
        cachedBytes += UInt64(span.length)
        cachedAllocationBytes += bytes
    }

    /// Background warm-up of one tensor a caller will acquire next (host-stage prefetch).
    /// Admits a miss only when it fits without eviction and is not excluded; never pins,
    /// counts no hit and returns whether the tensor is resident. The later `acquire`
    /// still validates source identity and counts its own hit.
    public func prefetch(_ name: String) throws -> (resident: Bool, read: Bool) {
        lock.lock()
        defer { lock.unlock() }
        let key = canonical(name)
        if entries[key] != nil { return (true, false) }
        let span = try geometry(key)
        let bytes = charge(span)
        guard !excluded(key), UInt64(span.length) <= maximumTensorBytes,
            bytes <= budget.limit - budget.allocated, entries.count < maximumEntries
        else { return (false, false) }
        try admit(key, span: span)
        return (true, true)
    }

    /// Evicts idle entries. Existing leases stay valid and charged; callers can
    /// construct a new cache with a different budget after outstanding work joins.
    public func trim() {
        lock.lock()
        defer { lock.unlock() }
        let idle = entries.compactMap { name, entry in entry.pinned ? nil : name }
        for name in idle { evict(name) }
    }
    public func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            byteLimit: budget.limit, allocatedBytes: budget.allocated, cachedBytes: cachedBytes,
            cachedAllocationBytes: cachedAllocationBytes, entries: entries.count,
            pinnedEntries: entries.values.filter(\.pinned).count,
            hits: hits, misses: misses, evictions: evictions, readBytes: readBytes, bypasses: bypasses,
            wiredBytes: wiredBytes, wireFallbacks: wireFallbacks)
    }
}
