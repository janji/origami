import Darwin
import Foundation
import Metal
import OrigamiEngine

private func residentRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try condition() else { throw EngineError(message) }
}
private func residentReject(_ body: () throws -> Void) throws {
    do { try body() } catch is EngineError { return }
    throw EngineError("Invalid resident operation unexpectedly succeeded")
}
private func residentDevice() throws -> MTLDevice {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal device") }
    return device
}
private final class ResidentFixture {
    let directory: URL, payload: URL
    let store: PackStore
    let data: Data
    init() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("origami-resident-" + UUID().uuidString)
        let payload = directory.appendingPathComponent("tensor.bin")
        let data = Data((0..<256).map { UInt8($0) })
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        try data.write(to: payload)
        var tensors: [String: Any] = [:]
        let names = [
            "a", "b", "c", "d", "model.layers.0.mlp.switch_mlp.gate_proj.weight",
            "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", "model.embed_tokens.weight",
        ]
        for (index, name) in names.enumerated() {
            tensors["language_model." + name] = [
                "file": "tensor.bin", "dtype": "U32", "offset": index * 4,
                "length": 4, "shape": [1],
            ]
        }
        tensors["language_model.bad-shape"] = ["file": "tensor.bin", "dtype": "U32", "offset": 0, "length": 4, "shape": [2]]
        tensors["language_model.bad-type"] = ["file": "tensor.bin", "dtype": "unknown", "offset": 0, "length": 4, "shape": [1]]
        tensors["language_model.overflow"] = ["file": "tensor.bin", "dtype": "U32", "offset": 0, "length": 4, "shape": [Int.max, 2]]
        store = try PackStore(model: ModelSource(fixture: directory, spans: tensors))
        self.directory = directory
        self.payload = payload
        self.data = data
    }
    deinit { try? FileManager.default.removeItem(at: directory) }
    func mutate() throws {
        let writer = try FileHandle(forWritingTo: payload)
        try writer.write(contentsOf: Data([0xff]))
        try writer.close()
    }
}

func residentTensorsPreserveBytesAndBoundLRUGroupAdmission() throws {
    let fixture = try ResidentFixture()
    let device = try residentDevice()
    let page = UInt64(getpagesize())
    let ledger = try MemoryLedger(ceiling: 20_000_000, transientMargin: 1)
    try autoreleasepool {
        let cache = try ResidentTensorCache(store: fixture.store, device: device, ledger: ledger, byteLimit: page * 2)
        try autoreleasepool {
            guard let first = try cache.acquire(["a", "language_model.b", "a"]) else { throw EngineError("Valid resident group bypassed") }
            try residentRequire(
                first.map(\.name) == ["language_model.a", "language_model.b", "language_model.a"], "Resident group order/aliases changed")
            try residentRequire(first[0].buffer === first[2].buffer, "Duplicate names allocated another buffer")
            for lease in first {
                let offset = Int(lease.span.offset)
                let actual = Data(bytes: lease.buffer.contents(), count: lease.length)
                try residentRequire(
                    actual == fixture.data.subdata(in: offset..<(offset + lease.length)),
                    "Resident tensor bytes were converted or misplaced")
            }
            try withExtendedLifetime(first) {
                try residentRequire(try cache.acquire("c") == nil, "Pinned tensors were evicted")
                cache.trim()
                try residentRequire(cache.snapshot().entries == 2 && cache.snapshot().pinnedEntries == 2, "Trim evicted a live lease")
            }
        }
        // Touch a after b so the next miss must evict b.
        try autoreleasepool { _ = try cache.acquire("a") }
        try autoreleasepool { _ = try cache.acquire("c") }
        let before = cache.snapshot()
        try autoreleasepool { _ = try cache.acquire("a") }
        try residentRequire(cache.snapshot().readBytes == before.readBytes, "LRU evicted the more recently used tensor")
        try autoreleasepool { _ = try cache.acquire("b") }
        let after = cache.snapshot()
        try residentRequire(
            after.evictions == 2 && after.readBytes == 16 && after.allocatedBytes <= page * 2,
            "Resident LRU/read accounting differs from actual loads")
        try residentRequire(try cache.acquire(["a", "b", "c"]) == nil, "Group larger than budget partially admitted")
        cache.trim()
        try residentRequire(
            cache.snapshot().allocatedBytes == 0 && ledger.snapshot().categories["resident-tensors"] == nil,
            "Unpinned resources stayed charged after trim")
    }
    try residentRequire(ledger.snapshot().reserved == 0, "Resident helper leaked its metadata or resources")
}

func residentTensorRawBufferEscapeRetainsPhysicalBudget() throws {
    let fixture = try ResidentFixture()
    let device = try residentDevice()
    let page = UInt64(getpagesize())
    let ledger = try MemoryLedger(ceiling: 20_000_000, transientMargin: 1)
    var escaped: MTLBuffer?
    try autoreleasepool {
        var cache: ResidentTensorCache? = try ResidentTensorCache(store: fixture.store, device: device, ledger: ledger, byteLimit: page)
        try autoreleasepool {
            var lease = try cache!.acquire("a")
            escaped = lease!.buffer
            lease = nil
            cache!.trim()
        }
        let state = cache!.snapshot()
        try residentRequire(state.entries == 0 && state.allocatedBytes == page, "Raw escape lost its physical allocation charge")
        try residentRequire(try cache!.acquire("b") == nil, "Escaped evicted resource allowed resident budget overrun")
        cache = nil
    }
    try residentRequire(
        escaped!.length == 4 && ledger.snapshot().categories["resident-tensors"] == page,
        "Destroying cache released a live raw Metal resource reservation")
    escaped = nil
    try residentRequire(ledger.snapshot().reserved == 0, "Final raw resource release did not retire reservations")
}

func residentTensorEligibilityAndGeometryAreExplicit() throws {
    let fixture = try ResidentFixture()
    let device = try residentDevice()
    let ledger = try MemoryLedger(ceiling: 20_000_000, transientMargin: 1)
    try autoreleasepool {
        let cache = try ResidentTensorCache(store: fixture.store, device: device, ledger: ledger, byteLimit: UInt64(getpagesize()) * 4)
        for name in [
            "model.layers.0.mlp.switch_mlp.gate_proj.weight",
            "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", "model.embed_tokens.weight",
        ] {
            try residentRequire(try cache.acquire(name) == nil, "Excluded whole tensor entered residency implicitly")
            try residentRequire(try cache.acquire(name, allowExcluded: true) != nil, "Explicit excluded tensor opt-in failed")
            let warm = cache.snapshot()
            try residentRequire(try cache.acquire(name) == nil, "Cached opt-in tensor bypassed the next call's exclusion policy")
            let refused = cache.snapshot()
            try residentRequire(
                refused.hits == warm.hits && refused.readBytes == warm.readBytes && refused.bypasses == warm.bypasses + 1,
                "Warm excluded refusal changed hit/read accounting")
            try residentRequire(try cache.acquire(name, allowExcluded: true) != nil, "Warm explicit opt-in failed")
            try residentRequire(
                cache.snapshot().hits == warm.hits + 1 && cache.snapshot().readBytes == warm.readBytes,
                "Warm explicit opt-in reread or failed to count the hit")
        }
        for name in ["missing", "bad-shape", "bad-type", "overflow"] { try residentReject { _ = try cache.acquire(name) } }
        try residentReject { _ = try cache.acquire([]) }
        try residentReject { _ = try cache.acquire(Array(repeating: "a", count: 17)) }
        try residentReject {
            _ = try ResidentTensorCache(store: fixture.store, device: device, ledger: ledger, byteLimit: 1, maximumTensorBytes: 400_000_001)
        }
        let tiny = try ResidentTensorCache(
            store: fixture.store, device: device, ledger: ledger, byteLimit: 1_000_000, maximumTensorBytes: 2)
        try residentRequire(try tiny.acquire("a") == nil && tiny.snapshot().readBytes == 0, "Individual tensor bound was ignored")
    }
    try residentRequire(ledger.snapshot().reserved == 0, "Eligibility fixtures retained resident resources")
}

func residentTensorSourceMutationNeverPublishesOrHits() throws {
    let device = try residentDevice()
    let page = UInt64(getpagesize())
    for warm in [false, true] {
        let fixture = try ResidentFixture()
        let ledger = try MemoryLedger(ceiling: 20_000_000, transientMargin: 1)
        try autoreleasepool {
            let cache = try ResidentTensorCache(store: fixture.store, device: device, ledger: ledger, byteLimit: page)
            if warm { try autoreleasepool { _ = try cache.acquire("a") } }
            try fixture.mutate()
            try residentReject { _ = try cache.acquire("a") }
            let state = cache.snapshot()
            try residentRequire(state.hits == 0 && state.entries == (warm ? 1 : 0), "Changed source published bytes or served a cache hit")
            if !warm {
                try residentRequire(state.readBytes == 0 && state.allocatedBytes == 0, "Failed read retained an unpublished allocation")
            }
        }
        try residentRequire(ledger.snapshot().reserved == 0, "Mutation failure leaked a resident reservation")
    }
}

/// The VM's user-wire count of the region holding `pointer` (0 when it holds none).
private func userWireCount(_ pointer: UnsafeMutableRawPointer) -> Int {
    var address = mach_vm_address_t(UInt(bitPattern: pointer))
    var size: mach_vm_size_t = 0
    var object: mach_port_t = 0
    var info = vm_region_basic_info_data_64_t()
    var count = mach_msg_type_number_t(MemoryLayout<vm_region_basic_info_data_64_t>.size / MemoryLayout<Int32>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: Int32.self, capacity: Int(count)) {
            mach_vm_region(mach_task_self_, &address, &size, VM_REGION_BASIC_INFO_64, $0, &count, &object)
        }
    }
    guard result == KERN_SUCCESS, address <= mach_vm_address_t(UInt(bitPattern: pointer)) else { return 0 }
    return Int(info.user_wired_count)
}

/// `wire_decode_slabs` wiring of both resident caches: a new entry is wired (the VM's count and the snapshot), an
/// evicted entry unwires while an escaped buffer still holds it, wiring off wires nothing, and the bytes are the same.
func residentCacheWiringFollowsEntriesAndKeepsBytes() throws {
    let device = try residentDevice()
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("origami-resident-wire-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let a = (0..<(16 * 1024)).map { Float($0 % 977) / 7 }
    let b = (0..<(16 * 1024)).map { BF16.encode(Float($0 % 613) / 5) }
    var data = a.withUnsafeBytes { Data($0) }
    data.append(b.withUnsafeBytes { Data($0) })
    try data.write(to: directory.appendingPathComponent("tensor.bin"))
    let store = try PackStore(
        model: ModelSource(
            fixture: directory,
            spans: [
                "language_model.model.w.a.weight": [
                    "file": "tensor.bin", "dtype": "F32", "offset": 0, "length": a.count * 4, "shape": [16, 1024],
                ],
                "language_model.model.w.b.weight": [
                    "file": "tensor.bin", "dtype": "BF16", "offset": a.count * 4, "length": b.count * 2, "shape": [16, 1024],
                ],
            ]))
    let ledger = try MemoryLedger(ceiling: 20_000_000, transientMargin: 1)
    var raw: [Bool: [Data]] = [:]
    var dense: [Bool: [Data]] = [:]
    for wire in [false, true] {
        try autoreleasepool {
            let cache = try ResidentTensorCache(
                store: store, device: device, ledger: ledger, byteLimit: UInt64(a.count * 4 + b.count * 2), wire: wire)
            var escaped: MTLBuffer?
            try autoreleasepool {
                let lease = try cache.acquire("model.w.a.weight")!
                escaped = lease.buffer
                raw[wire, default: []].append(Data(bytes: lease.buffer.contents(), count: lease.length))
                try residentRequire(
                    cache.snapshot().wiredBytes == (wire ? UInt64(a.count * 4) : 0) && cache.snapshot().wireFallbacks == 0
                        && (userWireCount(lease.buffer.contents()) > 0) == wire, "Resident entry wiring did not follow wire_decode_slabs")
            }
            cache.trim()
            try residentRequire(
                cache.snapshot().wiredBytes == 0 && userWireCount(escaped!.contents()) == 0, "An evicted resident entry stayed wired")
            try autoreleasepool {
                let lease = try cache.acquire("model.w.b.weight")!
                raw[wire, default: []].append(Data(bytes: lease.buffer.contents(), count: lease.length))
                try residentRequire(
                    cache.snapshot().evictions == 1 && cache.snapshot().wiredBytes == (wire ? UInt64(b.count * 2) : 0)
                        && (userWireCount(lease.buffer.contents()) > 0) == wire, "Resident wiring did not follow the new entry")
            }
            var float: ResidentFloatTensorCache? = try ResidentFloatTensorCache(
                store: store, device: device, ledger: ledger, byteLimit: UInt64(a.count * 8), wire: wire)
            var kept: MTLBuffer?
            try autoreleasepool {
                for name in ["model.w.a.weight", "model.w.b.weight"] {
                    let lease = try float!.acquire(name)!
                    kept = lease.buffer
                    dense[wire, default: []].append(Data(bytes: lease.buffer.contents(), count: lease.elementCount * 4))
                    try residentRequire(
                        (userWireCount(lease.buffer.contents()) > 0) == wire, "Dense resident wiring did not follow wire_decode_slabs")
                }
            }
            let state = float!.snapshot()
            try residentRequire(
                state.wiredBytes == (wire ? state.allocatedBytes : 0) && state.wireFallbacks == 0,
                "Dense resident wiring counters differ from its entries")
            float = nil
            try residentRequire(userWireCount(kept!.contents()) == 0, "A released dense resident matrix stayed wired")
        }
    }
    try residentRequire(raw[true] == raw[false] && dense[true] == dense[false], "Wiring changed resident bytes")
    try residentRequire(ledger.snapshot().reserved == 0, "Resident wiring fixture leaked reservations")
}
