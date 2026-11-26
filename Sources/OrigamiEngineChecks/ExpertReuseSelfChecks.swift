import CryptoKit
import Foundation
import Metal
import OrigamiEngine

/// Model-free checks of reuse admission: full-size slot ownership over a frozen route fixture, bypass and borrowed
/// slots under slow writers, cancellation and replay resets; every ledger charge is retired at the end.
package enum ExpertReuseSelfChecks {
    private static func require(_ value: @autoclosure () -> Bool, _ message: String) throws {
        guard value() else { throw EngineError(message) }
    }

    package static func actualCapacityAndLifetime() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
        let ledger = try MemoryLedger(ceiling: 16_000_000_000, transientMargin: 1_000_000_000)
        let fixtureBytes: UInt64 = 1_048_576
        let fixtureCharge = try ledger.reserve(bytes: fixtureBytes, category: "reuse-route-fixture")
        defer { try? ledger.release(fixtureCharge) }
        let fixture = CheckFixtures.file("routes/expert-reuse-routes.bin")
        let routes = try Data(contentsOf: fixture)
        try require(
            routes.count == 511 * 48 * 10 * 2
                && SHA256.hash(data: routes).map { String(format: "%02x", $0) }.joined()
                    == "7c840ef595bbc6c6c6469125146c09e7053522d2b7d21b73411601ecfda3eb67",
            "The expert reuse route fixture changed")
        func route(_ token: Int, _ layer: Int) -> [Int] {
            let start = (token * 48 + layer) * 20
            var ids: [Int] = []
            for rank in 0..<10 {
                let offset = start + rank * 2
                let low = Int(routes[offset])
                let high = Int(routes[offset + 1])
                ids.append(low + high * 256)
            }
            return ids
        }
        for capacity in [3424, 3192] {
            try autoreleasepool {
                let cache = try ExpertCache(
                    device: device, ledger: ledger,
                    byteLimit: UInt64(capacity * PackStore.expertBytes), reuseAdmission: true)
                try require(
                    ledger.snapshot().categories["expert-reuse-metadata"] == ExpertReusePolicy.admittedBytes,
                    "Reuse metadata was not charged")
                // Allocate the actual full slot geometry. Synthetic records
                // need only a marker; every full Metal allocation is charged.
                for i in 0..<capacity {
                    let key = try ExpertKey(layer: i / 512, expert: i % 512)
                    let a = try cache.acquire(key)
                    let bytes = try cache.beginRead(a.ticket)
                    bytes.bytes.baseAddress!.storeBytes(of: UInt32(i), as: UInt32.self)
                    try require(cache.completeRead(a.ticket, verified: true), "Synthetic producer failed")
                }
                try require(
                    cache.snapshot().allocatedBytes == UInt64(capacity * PackStore.expertBytes), "Actual-size cache was not allocated")
                try cache.setReuseDecode(true)
                for token in 0..<511 {
                    for layer in 0..<48 {
                        // Real frozen tuning routes exercise the calibrated
                        // reuse distribution; synthetic records still prove
                        // ownership without loading a model or expert weights.
                        let ids = route(token, layer)
                        let order = try cache.beginDecodeRow(layer: layer, experts: ids)
                        try require(Set(order) == Set(ids), "Admission ordering changed the true experts")
                        var tickets: [ExpertTicket] = []
                        for id in order {
                            let key = try ExpertKey(layer: layer, expert: id)
                            let a = try cache.acquire(key)
                            if a.newlyReserved {
                                let bytes = try cache.beginRead(a.ticket)
                                bytes.bytes.baseAddress!.storeBytes(of: UInt32(layer * 512 + id), as: UInt32.self)
                                try require(cache.completeRead(a.ticket, verified: true), "Demand producer failed")
                            }
                            let data = try cache.pin(a.ticket)
                            try require(
                                data.bytes.baseAddress!.load(as: UInt32.self) == UInt32(layer * 512 + id),
                                "Admission returned another expert")
                            tickets.append(a.ticket)
                        }
                        // Double consumer/host pin must keep a bypass record
                        // alive until its last owner, like SourceExpertLoader.
                        let extra = try cache.pin(tickets[0])
                        for ticket in tickets { try cache.unpinAfterGPUCompletion(ticket) }
                        do {
                            _ = try cache.resizeQuiescent(to: UInt64(capacity * PackStore.expertBytes))
                            throw EngineError("Busy resize admitted")
                        } catch let error as EngineError {
                            try require(error.code == "expert_phase_busy", "Busy resize changed failure contract")
                        }
                        _ = extra.bytes.baseAddress!.load(as: UInt32.self)
                        try cache.unpinAfterGPUCompletion(tickets[0])
                    }
                    if token == 3 { _ = try cache.resizeQuiescent(to: UInt64((capacity == 3424 ? 3360 : 2928) * PackStore.expertBytes)) }
                }
                let snapshot = cache.snapshot()
                try require(
                    snapshot.reuseRows == 511 * 48 && snapshot.reuseTransientSlots == 24,
                    "Reuse path/charged staging did not run")
                try require(
                    snapshot.pinnedSlots == 0 && snapshot.inFlightReads == 0 && snapshot.reservedSlots == 0,
                    "Reuse fixture retained a reader or pin")
                try require(
                    snapshot.reuseAdmissions! > 0,
                    "Admission fixture did not exercise ranked retention")
                // The frozen full-size replay has zero bypasses: new demands
                // outrank its coldest residents. Coverage must follow that
                // distribution, not invent admission rejections. A constrained
                // fixture below independently exercises bypass/slow writers.
                try cache.setReuseDecode(false)
                _ = try cache.resizeQuiescent(to: 0)
            }
            try require(ledger.snapshot().reserved == fixtureBytes, "Actual-size reuse owner leaked")
        }
        try boundedSlowWriters(device: device, ledger: ledger)
        try require(ledger.snapshot().reserved == fixtureBytes, "Slow-writer owner leaked")
    }

    private static func boundedSlowWriters(device: MTLDevice, ledger: MemoryLedger) throws {
        try autoreleasepool {
            let cache = try ExpertCache(
                device: device, ledger: ledger,
                byteLimit: UInt64(64 * PackStore.expertBytes), reuseAdmission: true)
            try cache.setReuseDecode(true)
            // Forty retained slots cannot hold the 480 repeatedly used experts.
            // The same frozen policy therefore exercises selective admission.
            for _ in 0..<32 {
                for layer in 0..<48 {
                    for id in try cache.beginDecodeRow(layer: layer, experts: Array(0..<10)) {
                        let a = try cache.acquire(ExpertKey(layer: layer, expert: id))
                        if a.newlyReserved {
                            let bytes = try cache.beginRead(a.ticket)
                            bytes.bytes.baseAddress!.storeBytes(of: UInt32(layer * 512 + id), as: UInt32.self)
                            try require(cache.completeRead(a.ticket, verified: true), "Constrained producer failed")
                        }
                        let data = try cache.pin(a.ticket)
                        try require(
                            data.bytes.baseAddress!.load(as: UInt32.self) == UInt32(layer * 512 + id),
                            "Constrained admission returned another expert")
                        try cache.unpinAfterGPUCompletion(a.ticket)
                    }
                }
            }
            try require(
                cache.snapshot().reuseBypasses! > 0 && cache.snapshot().reuseAdmissions! > 0,
                "Constrained admission did not discriminate retain/bypass")
            // Keep 28 primary writers alive after hypothetical replica wins.
            // The last four must borrow bounded residency, never live buffers.
            var held: [(ExpertTicket, ExpertBuffer, UInt32)] = []
            var bypasses = cache.snapshot().reuseBypasses!
            stress: for _ in 0..<32 {
                for layer in 0..<48 {
                    for id in try cache.beginDecodeRow(layer: layer, experts: Array(0..<10)) {
                        let a = try cache.acquire(ExpertKey(layer: layer, expert: id))
                        guard a.newlyReserved else { continue }
                        let bytes = try cache.beginRead(a.ticket)
                        let marker = UInt32(layer * 512 + id)
                        bytes.bytes.baseAddress!.storeBytes(of: marker, as: UInt32.self)
                        let now = cache.snapshot().reuseBypasses!
                        if now > bypasses {
                            held.append((a.ticket, bytes, marker))
                        } else {
                            try require(cache.completeRead(a.ticket, verified: true), "Stress producer failed")
                        }
                        bypasses = now
                        if held.count == 28 { break stress }
                    }
                }
            }
            try require(
                held.count == 28 && cache.snapshot().reuseBorrowedBypasses! > 0,
                "Slow-writer test did not borrow bounded residency")
            try require(
                cache.snapshot().allocatedBytes <= UInt64(64 * PackStore.expertBytes),
                "Slow-writer borrowing grew the pool")
            for (ticket, bytes, marker) in held {
                try require(bytes.bytes.baseAddress!.load(as: UInt32.self) == marker, "Live bypass writer was overwritten")
                try require(!cache.completeRead(ticket, verified: false), "Unverified slow writer published")
            }
            held.removeAll()
            try require(
                cache.snapshot().inFlightReads == 0 && cache.snapshot().pinnedSlots == 0,
                "Slow bypass producer did not join")
            try cache.setReuseDecode(false)
            _ = try cache.resizeQuiescent(to: 0)
        }
    }

    package static func cancellationAndMetadata() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
        let ledger = try MemoryLedger(ceiling: 250_000_000, transientMargin: 1_000_000)
        try autoreleasepool {
            let cache = try ExpertCache(
                device: device, ledger: ledger,
                byteLimit: UInt64(32 * PackStore.expertBytes), reuseAdmission: true)
            try cache.setReuseDecode(true)
            let ids = Array(0..<10)
            let order = try cache.beginDecodeRow(layer: 0, experts: ids)
            var active: [ExpertTicket] = []
            for id in order {
                let a = try cache.acquire(ExpertKey(layer: 0, expert: id))
                _ = try cache.beginRead(a.ticket)
                active.append(a.ticket)
            }
            for ticket in active { try require((try? cache.cancel(ticket)) == nil, "A live read destination was cancelled") }
            try require(cache.snapshot().inFlightReads == 10, "Cancellation recycled live destinations")
            for ticket in active {
                try require(!cache.completeRead(ticket, verified: false), "Unverified expert published")
                try require(!cache.completeRead(ticket, verified: true), "Late completion published")
            }
            try require(cache.snapshot().inFlightReads == 0 && cache.snapshot().activeSlots == 0, "Cancelled destinations not retired")
            try cache.setReuseDecode(false)
            try cache.setReuseDecode(true)
            _ = try cache.beginDecodeRow(layer: 0, experts: ids)
            _ = try cache.beginDecodeRow(layer: 1, experts: ids)
            // A staged next router can be discarded by the exact repair path.
            // Replay must remain legal and discard its uncertain history.
            _ = try cache.beginDecodeRow(layer: 0, experts: Array(ids.reversed()))
            try require(cache.snapshot().reuseReplayResets == 1, "Replay history did not reset")
            _ = try cache.beginDecodeRow(layer: 1, experts: ids)
        }
        try require(ledger.snapshot().reserved == 0, "Metadata/cancellation owner leaked")
    }

    /// 20,000 random cache states (phases, pins, bypass, held layers, protected rows, tied scores, empty and resized
    /// caches, rows between events): the free counters and per-layer victim minima give what today's full scans give.
    /// ROADMAP C5: a mixed-size GGUF's records in up to three slot sizes. The grouping of the GSQ-RCO IQ3_XXS layers;
    /// each record read into its class's own slot (its length, no slot overlapping another); the limit split by layers'
    /// bytes; a full class evicts only its own records; the GPU table points each key at its own slot's address; reuse
    /// decode and speculative admission stay in a key's class (`victimIndexMismatch` agrees); a cache too small for its
    /// classes keeps one slot size; every charge retired.
    package static func sizeClasses() throws {
        // The IQ3_XXS release's records by layer (INTERNALS.md): nine sizes, 23.7% padding in one class, 4.8% in three.
        let iq3xxs = [1_356_816, 1_459_216, 1_561_616, 1_766_416, 1_868_816, 1_920_016, 1_971_216, 2_176_016, 2_329_616]
        let counts = [8, 6, 6, 4, 4, 7, 4, 3, 6]
        let layerBytes = zip(iq3xxs, counts).flatMap { Array(repeating: $0, count: $1) }.shuffled()
        let grouped = ExpertCache.SizeClasses.grouping(layerBytes)
        try require(grouped.recordBytes == [1_561_616, 1_971_216, 2_329_616], "IQ3_XXS grouping \(grouped.recordBytes)")
        try require(
            zip(layerBytes, grouped.layerClasses).allSatisfy { size, c in
                size <= grouped.recordBytes[c] && (c == 0 || size > grouped.recordBytes[c - 1])
            }, "A layer is not in the smallest class holding it")
        try require(ExpertCache.SizeClasses.grouping(Array(repeating: 2_329_616, count: 48)) == .one, "One size made classes")
        try require(
            ExpertCache.SizeClasses.grouping([1_000_000, 2_000_000] + Array(repeating: 2_000_000, count: 46)).recordBytes
                == [1_000_000, 2_000_000], "Two sizes")

        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
        let record = PackStore.expertBytes
        let sizes = ExpertCache.SizeClasses(
            recordBytes: [record / 2 / 16 * 16, record * 3 / 4 / 16 * 16, record],
            layerClasses: (0..<48).map { $0 % 3 == 0 ? 0 : $0 % 3 == 1 ? 1 : 2 })
        let ledger = try MemoryLedger(ceiling: UInt64(400 * record), transientMargin: 1_000_000)
        try autoreleasepool {
            // 240 full-size slots' bytes: each class gets about a third by bytes, more records in the smaller ones.
            let limit = UInt64(240 * record)
            let cache = try ExpertCache(
                device: device, ledger: ledger, byteLimit: limit, maximumByteLimit: UInt64(320 * record), reuseAdmission: true,
                sizes: sizes, minimumClassRecords: 40)
            try require(cache.sizes == sizes, "The classes were not used")
            let split = cache.snapshot().sizeClasses ?? []
            try require(
                split.count == 3 && split.map(\.byteLimit).reduce(0, +) <= limit
                    && zip(split, sizes.recordBytes).allSatisfy { $0.recordBytes == $1 && $0.byteLimit % UInt64(8 * $1) == 0 },
                "Limits were not split in whole slabs: \(split)")
            try require(
                split.map { Int($0.byteLimit) / $0.recordBytes } == (0..<3).map { cache.slotLimit(sizeClass: $0) }
                    && cache.slotLimit == split.map({ Int($0.byteLimit) / $0.recordBytes }).min(),
                "Slot limits differ")
            // Fill every class to its limit, then one more record per class.
            var spans: [(buffer: ObjectIdentifier, range: Range<Int>)] = []
            var loaded: [Int: [ExpertKey]] = [:]
            for c in 0..<3 {
                let layers = (0..<48).filter { sizes.layerClasses[$0] == c }
                for i in 0..<(cache.slotLimit(sizeClass: c) + 1) {
                    let key = try ExpertKey(layer: layers[i % layers.count], expert: i / layers.count)
                    let a = try cache.acquire(key)
                    try require(a.newlyReserved, "A new key was resident")
                    let buffer = try cache.beginRead(a.ticket)
                    try require(buffer.length == sizes.recordBytes[c] && cache.recordBytes(key) == buffer.length, "Wrong slot length")
                    buffer.bytes.initializeMemory(as: UInt8.self, repeating: UInt8(truncatingIfNeeded: c * 64 + i))
                    try require(cache.completeRead(a.ticket, verified: true), "Read did not complete")
                    spans.append((ObjectIdentifier(buffer.buffer), buffer.offset..<(buffer.offset + buffer.length)))
                    loaded[c, default: []].append(key)
                }
            }
            let snapshot = cache.snapshot()
            try require(snapshot.evictions == 3, "Each full class should have evicted one record: \(snapshot.evictions)")
            try require(snapshot.allocatedBytes <= limit, "Allocated past the limit")
            // The newest record of each class is resident; its bytes are its own (no other slot overlapped it).
            for c in 0..<3 {
                let key = loaded[c]!.last!
                let a = try cache.acquire(key)
                try require(!a.newlyReserved, "Class \(c)'s last record was evicted")
                let buffer = try cache.pin(a.ticket)
                let marker = UInt8(truncatingIfNeeded: c * 64 + loaded[c]!.count - 1)
                try require(buffer.bytes.allSatisfy { $0 == marker }, "Class \(c)'s record was overwritten")
                try cache.unpinAfterGPUCompletion(a.ticket)
            }
            // The GPU table: each resident key at its own slot's address.
            let table = try cache.gpuTable()
            var addresses: Set<UInt64> = []
            for c in 0..<3 {
                for key in loaded[c]!.suffix(20) {
                    let entry = table.keys.contents().load(fromByteOffset: (key.layer * 512 + key.expert) * 8, as: UInt64.self)
                    guard entry & 0xFFFF_FFFF != 0 else { continue }  // transient bypass slots are not published
                    let a = try cache.acquire(key)
                    let buffer = try cache.pin(a.ticket)
                    let address = table.slots.contents().load(fromByteOffset: Int((entry & 0xFFFF_FFFF) - 1) * 8, as: UInt64.self)
                    try require(address == buffer.buffer.gpuAddress + UInt64(buffer.offset), "GPU table address differs")
                    try require(addresses.insert(address).inserted, "Two keys share a GPU slot")
                    try cache.unpinAfterGPUCompletion(a.ticket)
                }
            }
            try require(addresses.count > 30, "Too few published records: \(addresses.count)")
            // Reuse decode and speculation stay in a key's class; the indexed victims match full scans.
            try cache.setReuseDecode(true)
            var random = SystemRandomNumberGenerator()
            for step in 0..<3_000 {
                let layer = step % 48
                _ = try cache.beginDecodeRow(layer: layer, experts: Array(Array(0..<40).shuffled(using: &random).prefix(10)))
                let key = try ExpertKey(layer: layer, expert: Int.random(in: 0..<40, using: &random))
                if let mismatch = cache.victimIndexMismatch(key) { throw EngineError("Class victim index: \(mismatch)") }
                if Bool.random(using: &random) {
                    let a = try cache.acquire(key)
                    if a.newlyReserved {
                        let buffer = try cache.beginRead(a.ticket)
                        try require(buffer.length == sizes.recordBytes[sizes.layerClasses[layer]], "Reuse slot of another class")
                        _ = cache.completeRead(a.ticket, verified: true)
                    }
                } else if case .admitted(let a) = try cache.acquireSpeculative(key) {
                    let buffer = try cache.beginRead(a.ticket)
                    try require(buffer.length == sizes.recordBytes[sizes.layerClasses[layer]], "Speculative slot of another class")
                    _ = cache.completeRead(a.ticket, verified: true)
                }
            }
            try cache.setReuseDecode(false)
            // A smaller limit retires slabs of every class, each within its new share.
            _ = try cache.resizeQuiescent(to: UInt64(120 * record))
            let after = cache.snapshot()
            try require(
                after.allocatedBytes <= UInt64(120 * record)
                    && zip(after.sizeClasses ?? [], split).allSatisfy { $0.byteLimit < $1.byteLimit },
                "Resize did not shrink every class: \(after.sizeClasses ?? [])")
            // Too small for three classes of 40 records: one slot size.
            let small = try ExpertCache(
                device: device, ledger: ledger, byteLimit: UInt64(64 * record), sizes: sizes, minimumClassRecords: 40)
            try require(small.sizes == .one && small.snapshot().sizeClasses == nil, "A small cache kept classes")
        }
        try require(ledger.reservedBytes == 0, "Size-class cache charges were not retired: \(ledger.reservedBytes)")
    }

    package static func victimIndex() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Metal unavailable") }
        let ledger = try MemoryLedger(ceiling: UInt64(128 * PackStore.expertBytes), transientMargin: 1_000_000)
        try autoreleasepool {
            let slots = 48
            let cache = try ExpertCache(
                device: device, ledger: ledger, byteLimit: UInt64(slots * PackStore.expertBytes),
                reuseAdmission: true)
            var random = SystemRandomNumberGenerator()
            func key() throws -> ExpertKey {
                try ExpertKey(layer: Int.random(in: 0..<48, using: &random), expert: Int.random(in: 0..<6, using: &random))
            }
            var reserved: [ExpertTicket] = []
            var reading: [ExpertTicket] = []
            var pinned: [ExpertTicket] = []
            var layer = 0
            var reuse = true
            let empty = try key()
            try require(cache.victimIndexMismatch(empty) == nil, "Empty cache victim differs")
            try cache.setReuseDecode(true)
            for state in 0..<20_000 {
                switch Int.random(in: 0..<100, using: &random) {
                case 0..<12:
                    layer = Int.random(in: 0..<20, using: &random) == 0 ? Int.random(in: 0..<48, using: &random) : (layer + 1) % 48
                    _ = try cache.beginDecodeRow(
                        layer: layer, experts: Array(Array(0..<6).shuffled(using: &random).prefix(Int.random(in: 1...6, using: &random))))
                case 12..<30:
                    if let a = try? cache.acquire(key()) {
                        if a.newlyReserved {
                            reserved.append(a.ticket)
                        } else if [.verified, .evictable, .pinned].contains(a.phase), pinned.count < 30 {
                            _ = try cache.pin(a.ticket)
                            pinned.append(a.ticket)
                        }
                    }
                case 30..<45:
                    switch try cache.acquireSpeculative(key(), overrideScore: Bool.random(using: &random)) {
                    case .admitted(let a), .overridden(let a): reserved.append(a.ticket)
                    default: break
                    }
                case 45..<60 where !reserved.isEmpty:
                    let ticket = reserved.remove(at: Int.random(in: reserved.indices, using: &random))
                    if Int.random(in: 0..<8, using: &random) == 0 {
                        try cache.cancel(ticket)
                    } else {
                        _ = try cache.beginRead(ticket)
                        reading.append(ticket)
                    }
                case 60..<75 where !reading.isEmpty:
                    let ticket = reading.remove(at: Int.random(in: reading.indices, using: &random))
                    _ = cache.completeRead(ticket, verified: Int.random(in: 0..<10, using: &random) != 0)
                case 75..<88 where !pinned.isEmpty:
                    try cache.unpinAfterGPUCompletion(pinned.remove(at: Int.random(in: pinned.indices, using: &random)))
                case 88..<93:
                    if Bool.random(using: &random) {
                        _ = cache.holdForGPU(layer: Int.random(in: 0..<48, using: &random))
                    } else {
                        cache.releaseForGPU(layer: Bool.random(using: &random) ? Int.random(in: 0..<48, using: &random) : nil)
                    }
                case 93..<96:
                    _ = try cache.resize(to: UInt64([0, 32, 40, 48].randomElement(using: &random)! * PackStore.expertBytes))
                case 96:
                    _ = try cache.gpuTable()
                case 97:
                    // Join every reader and pin, then switch phase: prefill (CLOCK) states check the any-index counter.
                    for ticket in reserved { try cache.cancel(ticket) }
                    for ticket in reading { _ = cache.completeRead(ticket, verified: false) }
                    for ticket in pinned { try cache.unpinAfterGPUCompletion(ticket) }
                    reserved.removeAll()
                    reading.removeAll()
                    pinned.removeAll()
                    reuse.toggle()
                    try cache.setReuseDecode(reuse)
                    cache.setPrefillScan(!reuse)
                default: break
                }
                if let mismatch = cache.victimIndexMismatch(try key()) {
                    throw EngineError("Indexed victim differs from the full scan at state \(state): \(mismatch)")
                }
            }
            for ticket in reserved { try cache.cancel(ticket) }
            for ticket in reading { _ = cache.completeRead(ticket, verified: false) }
            for ticket in pinned { try cache.unpinAfterGPUCompletion(ticket) }
            cache.releaseForGPU(layer: nil)
            try cache.setReuseDecode(false)
            let idle = try key()
            try require(cache.victimIndexMismatch(idle) == nil, "Idle cache victim differs")
            _ = try cache.resizeQuiescent(to: 0)
        }
        try require(ledger.snapshot().reserved == 0, "Victim index check leaked slabs")
    }
}
