import Darwin
import Foundation
import Metal
import ObjectiveC

/// Ledger reservation tied to one scratch buffer (an associated object), released when the buffer is freed.
private final class GatheredExpertAllocation {
    let ledger: MemoryLedger, reservation: MemoryLedger.Reservation
    init(_ ledger: MemoryLedger, bytes: UInt64) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: "gathered-expert-decode")
    }
    deinit { try? ledger.release(reservation) }
}
private enum GatheredExpertAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// Decode MoE for one token: the ten routed experts' gate, up, SiLU product and down read straight from their cache
/// records, then the canonical ten-way weighted reduction. Records are addressed through a ten-pointer argument buffer
/// (tier 2); `ranks[i]` places record i's outputs in its canonical-rank span, so any split of the ten experts into
/// groups writes the same bytes. Every projection keeps the pinned one-row QMV lane schedule: BITWISE equal to the
/// per-expert fused path. No weight repack, QMM selection or expert omission. Callers hold all ten cache pins until
/// the command joins; a Record alone does not stop a recyclable slot from being reused.
/// GGUF experts (`ModelLayout.ggufExperts`) take the same entry points on ggml's kernels (`GGUFExpertKernels`): the same
/// rank spans and reduction, f32 products; the queued (pre-encoded) group runs ggml's kernels on GPU-written addresses.
public final class GatheredExpertDecode {
    /// One expert record: its Metal buffer and byte offset (4-byte aligned, `PackStore.expertBytes` long, or a GGUF
    /// layer's own bytes in a smaller size class's slot).
    public struct Record {
        public let buffer: MTLBuffer
        public let offset: Int
        public init(buffer: MTLBuffer, offset: Int) {
            self.buffer = buffer
            self.offset = offset
        }
    }
    /// A caller-owned decode workspace: FP32 gate and up [10][640] (only `evaluate`'s separate dispatches use them),
    /// BF16 product [10][640], FP32 down [10][2560], the ten-record argument buffer and the UInt32 rank map.
    package struct RegionBuffers {
        package let gate, up, product, down, descriptors, ranks: MTLBuffer
        package init(gate: MTLBuffer, up: MTLBuffer, product: MTLBuffer, down: MTLBuffer, descriptors: MTLBuffer, ranks: MTLBuffer) {
            (self.gate, self.up, self.product, self.down, self.descriptors, self.ranks) = (gate, up, product, down, descriptors, ranks)
        }
        var all: [MTLBuffer] { [gate, up, product, down, descriptors, ranks] }
    }
    package var regionArgumentBufferBytes: Int { arguments.encodedLength }
    /// Counters and timings of the joined paths (`evaluate` and the check helpers); the encoded paths do not count.
    public struct Snapshot: Codable, Sendable {
        public let submittedCommands, completedCommands, submittedExperts, bufferUploadBytes, floatReadbackBytes: UInt64
        public let hostPrepareSeconds, hostJoinSeconds, gpuSeconds, demandPinJoinSeconds: Double
        public let retainedBytes, argumentBufferBytes: Int
        public let argumentBufferTier: Int
        /// Optional so older diagnostic observations remain decodable.
        public let sharedDuringDemandCalls: UInt64?
        public let sharedDuringDemandSeconds: Double?
    }
    private let device: MTLDevice, queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState, arguments: MTLArgumentEncoder
    private let tiledProduct, tiledDown: MTLComputePipelineState
    /// Queued-experts decode (`decode_preencoded_layers`): fills a deferred group on the GPU from the router's ranks and
    /// the cache's GPU residency table.
    private let residentLookup: MTLComputePipelineState
    private let ledger: MemoryLedger, fused: FusedExpertExecution, reductions: PinnedReductions
    private var submitted: UInt64 = 0, completed: UInt64 = 0, uploads: UInt64 = 0, readbacks: UInt64 = 0
    private var prepareSeconds = 0.0, joinSeconds = 0.0, gpuSeconds = 0.0, demandSeconds = 0.0
    private var sharedDemandCalls: UInt64 = 0
    private var sharedDemandSeconds = 0.0
    private var active = false
    private struct Scratch {
        let input, gate, up, product, down, weights, output, descriptors, ranks: MTLBuffer
        var all: [MTLBuffer] { [input, gate, up, product, down, weights, output, descriptors, ranks] }
    }
    private var scratch: Scratch?

    public init(ledger: MemoryLedger, fused: FusedExpertExecution, reductions: PinnedReductions) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
            device.argumentBuffersSupport == .tier2
        else {
            throw EngineError("Gathered expert decode requires Metal argument-buffer tier2")
        }
        try ModelContract.validate(device: device)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        let library = try device.makeLibrary(
            source: ModelLayout.expertMetalDefines + ReferenceKernels.quantizedBF16Preamble + Self.source
                + GatheredExpertTiledKernel.source, options: options)
        guard let function = library.makeFunction(name: "expert_gathered_qmv10") else { throw EngineError("Missing gathered QMV") }
        pipeline = try device.makeComputePipelineState(function: function)
        guard let tiledProductFunction = library.makeFunction(name: "expert_gate_up_product"),
            let tiledDownFunction = library.makeFunction(name: "expert_down_tile8")
        else {
            throw EngineError("Missing gathered tile8 kernels")
        }
        tiledProduct = try device.makeComputePipelineState(function: tiledProductFunction)
        tiledDown = try device.makeComputePipelineState(function: tiledDownFunction)
        guard let lookupFunction = library.makeFunction(name: "expert_resident_group") else {
            throw EngineError("Missing resident group lookup")
        }
        residentLookup = try device.makeComputePipelineState(function: lookupFunction)
        guard [tiledProduct, tiledDown].allSatisfy({ $0.threadExecutionWidth == 32 && $0.maxTotalThreadsPerThreadgroup >= 128 }) else {
            throw EngineError("Gathered tile8 resource limit differs")
        }
        guard pipeline.threadExecutionWidth == 32, pipeline.maxTotalThreadsPerThreadgroup >= 64 else {
            throw EngineError("Gathered QMV cannot preserve the two-SIMD threadgroup")
        }
        arguments = function.makeArgumentEncoder(bufferIndex: 1)
        guard tiledProductFunction.makeArgumentEncoder(bufferIndex: 1).encodedLength == arguments.encodedLength,
            tiledDownFunction.makeArgumentEncoder(bufferIndex: 1).encodedLength == arguments.encodedLength
        else {
            throw EngineError("Gathered tile8 descriptor layout differs")
        }
        guard arguments.encodedLength > 0, arguments.encodedLength <= 16384 else { throw EngineError("Gather descriptor exceeds bound") }
        // The resident lookup writes the ten record pointers as raw GPU addresses.
        guard arguments.encodedLength == 80 else { throw EngineError("Gathered descriptors are not ten GPU addresses") }
        self.device = device
        self.queue = queue
        self.ledger = ledger
        self.fused = fused
        self.reductions = reductions
    }
    public func snapshot() -> Snapshot {
        Snapshot(
            submittedCommands: submitted, completedCommands: completed, submittedExperts: submitted * 10,
            bufferUploadBytes: uploads, floatReadbackBytes: readbacks, hostPrepareSeconds: prepareSeconds,
            hostJoinSeconds: joinSeconds, gpuSeconds: gpuSeconds, demandPinJoinSeconds: demandSeconds,
            retainedBytes: scratch?.all.reduce(0, { $0 + $1.allocatedSize }) ?? 0,
            argumentBufferBytes: arguments.encodedLength, argumentBufferTier: 2,
            sharedDuringDemandCalls: sharedDemandCalls, sharedDuringDemandSeconds: sharedDemandSeconds)
    }
    func recordDemandPinJoin(seconds: Double) { demandSeconds += seconds }
    func recordSharedDuringDemand(seconds: Double) {
        sharedDemandCalls += 1
        sharedDemandSeconds += seconds
    }
    public func trim() throws {
        guard !active else { throw EngineError("Cannot trim an active gathered expert scope") }
        scratch = nil
    }
    private func allocate(_ bytes: Int) throws -> MTLBuffer {
        let page = Int(getpagesize())
        let charged = (bytes + page - 1) / page * page
        let owner = try GatheredExpertAllocation(ledger, bytes: UInt64(charged))
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared), buffer.allocatedSize <= charged else {
            throw EngineError("Gathered expert allocation exceeds admitted storage", code: "insufficient_memory")
        }
        objc_setAssociatedObject(buffer, &GatheredExpertAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        return buffer
    }
    private func validate(_ records: [Record], _ ranks: [Int], partial: Bool = false) throws {
        guard partial ? (1...10).contains(records.count) : records.count == 10,
            ranks.count == records.count, Set(ranks).count == ranks.count,
            ranks.allSatisfy({ (0..<10).contains($0) }),
            records.allSatisfy({
                $0.buffer.device.registryID == device.registryID && $0.offset >= 0 && $0.offset % 4 == 0 && $0.offset <= $0.buffer.length
                    && ModelLayout.smallestExpertRecordBytes <= $0.buffer.length - $0.offset
            })
        else { throw EngineError("Gathered expert records/rank permutation invalid") }
    }
    private func bind(_ records: [Record], _ ranks: [Int], descriptors: MTLBuffer, map: MTLBuffer) {
        arguments.setArgumentBuffer(descriptors, offset: 0)
        defer { arguments.setArgumentBuffer(nil, offset: 0) }
        for index in 0..<10 {
            arguments.setBuffer(
                index < records.count ? records[index].buffer : nil,
                offset: index < records.count ? records[index].offset : 0, index: index)
        }
        let values = map.contents().assumingMemoryBound(to: UInt32.self)
        for i in ranks.indices { values[i] = UInt32(ranks[i]) }
    }
    private func encode(
        _ command: MTLCommandBuffer, input: MTLBuffer, descriptors: MTLBuffer, ranks: MTLBuffer,
        records: [Record], kind: ExpertProjection, output: MTLBuffer
    ) throws {
        guard let e = command.makeComputeCommandEncoder() else { throw EngineError("Gathered QMV encoder unavailable") }
        e.setComputePipelineState(pipeline)
        e.setBuffer(input, offset: 0, index: 0)
        e.setBuffer(descriptors, offset: 0, index: 1)
        e.setBuffer(ranks, offset: 0, index: 2)
        e.setBuffer(output, offset: 0, index: 3)
        let shape = [
            kind.width, kind.outputs, kind.weightOffset, kind.scaleOffset, kind.biasOffset,
            kind == .down ? 8 : 16, kind == .down ? kind.width : 0,
        ].map(UInt32.init)
        shape.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 4) }
        for record in records { e.useResource(record.buffer, usage: .read) }
        e.dispatchThreads(
            MTLSize(width: kind.outputs * 32, height: records.count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        e.endEncoding()
    }
    private func joined(_ command: MTLCommandBuffer, shouldContinue: () -> Bool) throws {
        guard shouldContinue() else { throw IOOperationCancelled() }
        submitted += 1
        let start = DispatchTime.now().uptimeNanoseconds
        command.commit()
        try command.waitBounded()
        joinSeconds += Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
        if command.gpuStartTime > 0, command.gpuEndTime >= command.gpuStartTime {
            gpuSeconds += command.gpuEndTime - command.gpuStartTime
        }
        guard command.status == .completed else { throw EngineError.gpu("Gathered command failed after join", command.error) }
        completed += 1
        guard shouldContinue() else { throw IOOperationCancelled() }
    }
    /// Tile8 gate/up/product then down for `records`: eight output rows share activation work. Each record writes its
    /// own canonical-rank product/down span; the reduction runs once, after all ten ranks are written. `evaluate`
    /// keeps the separate QMV, product and down dispatches as the independent schedule.
    private func encodeTiledGroup(
        _ command: MTLCommandBuffer, input: MTLBuffer,
        records: [Record], ranks: [Int], buffers s: RegionBuffers
    ) throws {
        if let gguf = fused.gguf, let layout = ModelLayout.ggufExperts {
            // The f32 input row goes where the BF16 products would be; the products stay f32 in `gate`.
            return try gguf.encodeGroup(
                command, input: input, records: records.map { ($0.buffer, $0.offset) }, ranks: ranks,
                floats: s.product, gate: s.gate, up: s.up, down: s.down, layout: layout)
        }
        guard let gate = command.makeComputeCommandEncoder() else { throw EngineError("Gathered tile8 product encoder unavailable") }
        gate.setComputePipelineState(tiledProduct)
        for (i, buffer) in [input, s.descriptors, s.ranks, s.product, fused.decodeRegionSiLUTable].enumerated() {
            gate.setBuffer(buffer, offset: 0, index: i)
        }
        for record in records { gate.useResource(record.buffer, usage: .read) }
        gate.dispatchThreads(
            MTLSize(width: 640 / 8 * 32, height: records.count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        gate.endEncoding()
        guard let down = command.makeComputeCommandEncoder() else { throw EngineError("Gathered tile8 down encoder unavailable") }
        down.setComputePipelineState(tiledDown)
        for (i, buffer) in [s.product, s.descriptors, s.ranks, s.down].enumerated() { down.setBuffer(buffer, offset: 0, index: i) }
        for record in records { down.useResource(record.buffer, usage: .read) }
        down.dispatchThreads(
            MTLSize(width: 2560 / 8 * 32, height: records.count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        down.endEncoding()
    }
    /// Unchanged ten-expert grids/product/canonical reducer within a caller's
    /// final layer command. The consumeTen owner must retain all ten pins until
    /// that command joins, including any encoding, cancellation or GPU failure.
    package func encodeRegion(
        _ command: MTLCommandBuffer, inputBF16: MTLBuffer, weights: MTLBuffer,
        records: [Record], canonicalRanks: [Int], output: MTLBuffer,
        buffers s: RegionBuffers
    ) throws {
        try validate(records, canonicalRanks)
        let all = s.all + [output]
        let sizes = [
            10 * 640 * 4, 10 * 640 * 4, 10 * 640 * 2,
            10 * 2560 * 4, arguments.encodedLength, 10 * 4, 2560 * 4,
        ]
        let reads = [inputBF16, weights] + records.map(\.buffer)
        guard !active, inputBF16.device.registryID == device.registryID, inputBF16.length >= 2560 * 2,
            weights.device.registryID == device.registryID, weights.length >= 10 * 4,
            zip(all, sizes).allSatisfy({ $0.0.device.registryID == device.registryID && $0.0.length >= $0.1 }),
            Set(all.map { ObjectIdentifier($0) }).count == all.count,
            !all.contains(where: { scratch in reads.contains(where: { $0 === scratch }) }),
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued
        else {
            throw EngineError("Invalid caller-owned gathered expert region")
        }
        bind(records, canonicalRanks, descriptors: s.descriptors, map: s.ranks)
        try encodeTiledGroup(command, input: inputBF16, records: records, ranks: canonicalRanks, buffers: s)
        try reductions.encodeWeightedExpertReduce10(command, input: s.down, weights: weights, output: output, tokens: 1)
    }

    /// A group's region: every buffer on this device and long enough, none shared with another or with `reads`
    /// (the input, at least 2560 BF16 values, comes first).
    private func fits(_ s: RegionBuffers, reads: [MTLBuffer]) -> Bool {
        let sizes = [10 * 640 * 4, 10 * 640 * 4, 10 * 640 * 2, 10 * 2560 * 4, arguments.encodedLength, 40]
        return reads[0].length >= 5120 && reads.allSatisfy({ $0.device.registryID == device.registryID })
            && zip(s.all, sizes).allSatisfy({ $0.0.device.registryID == device.registryID && $0.0.length >= $0.1 })
            && Set(s.all.map { ObjectIdentifier($0) }).count == s.all.count
            && !s.all.contains(where: { scratch in reads.contains(where: { $0 === scratch }) })
    }
    /// Compute a nonempty subset into disjoint canonical-rank spans. The next
    /// group may reuse descriptors only AFTER this command joins. Reduction is
    /// deliberately separate and occurs once after all ten ranks are written.
    func encodeReadyGroup(
        _ command: MTLCommandBuffer, inputBF16: MTLBuffer,
        records: [Record], canonicalRanks: [Int], buffers s: RegionBuffers
    ) throws {
        try validate(records, canonicalRanks, partial: true)
        guard !active, fits(s, reads: [inputBF16] + records.map(\.buffer)), command.commandQueue.device.registryID == device.registryID,
            command.status == .notEnqueued
        else {
            throw EngineError("Invalid resident-first expert group")
        }
        bind(records, canonicalRanks, descriptors: s.descriptors, map: s.ranks)
        try encodeTiledGroup(command, input: inputBF16, records: records, ranks: canonicalRanks, buffers: s)
    }
    /// Queued-experts decode: the tiled group encoded before its records are known. Its grid sizes are read from `grids` at `offset`
    /// (product, then down: two (x, y, z) UInt32 triples) and its records from the descriptors, both written by
    /// `fillDeferredGroup` before the host signals the wait in front of it. The record buffers are made resident by the
    /// caller's residency set, since they are unknown here. Same kernels and grid as `encodeReadyGroup`. GGUF experts run
    /// `layer`'s kernels on the same indirect grids (`GGUFExpertKernels.encodeDeferredGroup`).
    static let deferredGridBytes = 24
    func encodeDeferredGroup(
        _ command: MTLCommandBuffer, layer: Int, inputBF16: MTLBuffer, buffers s: RegionBuffers,
        grids: MTLBuffer, offset: Int
    ) throws {
        guard !active, fits(s, reads: [inputBF16, grids]), offset >= 0, offset + Self.deferredGridBytes <= grids.length,
            (0..<ModelContract.layers).contains(layer),
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued
        else {
            throw EngineError("Invalid deferred expert group")
        }
        if let gguf = fused.gguf, let layout = ModelLayout.ggufExperts {
            return try gguf.encodeDeferredGroup(
                command, layer: layout.layers[layer], input: inputBF16, records: s.descriptors, ranks: s.ranks,
                grids: grids, offset: offset, floats: s.product, gate: s.gate, up: s.up, down: s.down)
        }
        guard let gate = command.makeComputeCommandEncoder() else { throw EngineError("Deferred product encoder unavailable") }
        gate.setComputePipelineState(tiledProduct)
        for (i, buffer) in [inputBF16, s.descriptors, s.ranks, s.product, fused.decodeRegionSiLUTable].enumerated() {
            gate.setBuffer(buffer, offset: 0, index: i)
        }
        gate.dispatchThreadgroups(
            indirectBuffer: grids, indirectBufferOffset: offset,
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        gate.endEncoding()
        guard let down = command.makeComputeCommandEncoder() else { throw EngineError("Deferred down encoder unavailable") }
        down.setComputePipelineState(tiledDown)
        for (i, buffer) in [s.product, s.descriptors, s.ranks, s.down].enumerated() { down.setBuffer(buffer, offset: 0, index: i) }
        down.dispatchThreadgroups(
            indirectBuffer: grids, indirectBufferOffset: offset + 12,
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        down.endEncoding()
    }
    /// Queued-experts decode: fills the deferred group at `offset` on the GPU with the routed experts (`pairs`, the router's
    /// rank-ordered (score, id) pairs) whose records `table` holds from before `epoch`; the group then runs without the host.
    func encodeResidentLookup(
        _ command: MTLCommandBuffer, pairs: MTLBuffer, table: (keys: MTLBuffer, slots: MTLBuffer),
        layer: Int, epoch: UInt32, buffers s: RegionBuffers, grids: MTLBuffer, offset: Int
    ) throws {
        // The kernel reads keys[layer * 512 + id] and writes ten descriptors (80 bytes) and ten ranks.
        guard (0..<48).contains(layer), pairs.length >= 80, table.keys.length >= 48 * 512 * 8, s.descriptors.length >= 80,
            s.ranks.length >= 40, offset >= 0, offset + Self.deferredGridBytes <= grids.length,
            [pairs, table.keys, table.slots, grids, s.descriptors, s.ranks].allSatisfy({ $0.device.registryID == device.registryID }),
            command.commandQueue.device.registryID == device.registryID,
            command.status == .notEnqueued, let e = command.makeComputeCommandEncoder()
        else {
            throw EngineError("Invalid resident group lookup")
        }
        e.setComputePipelineState(residentLookup)
        for (i, buffer) in [pairs, table.keys, table.slots].enumerated() { e.setBuffer(buffer, offset: 0, index: i) }
        var arguments = SIMD2<UInt32>(UInt32(layer), epoch)
        e.setBytes(&arguments, length: 8, index: 3)
        e.setBuffer(s.descriptors, offset: 0, index: 4)
        e.setBuffer(s.ranks, offset: 0, index: 5)
        e.setBuffer(grids, offset: offset, index: 6)
        e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        e.endEncoding()
    }
    /// Writes a deferred group's records, ranks and grid; zero records leave both dispatches empty.
    func fillDeferredGroup(records: [Record], canonicalRanks: [Int], buffers s: RegionBuffers, grids: MTLBuffer, offset: Int) throws {
        if !records.isEmpty {
            try validate(records, canonicalRanks, partial: true)
            bind(records, canonicalRanks, descriptors: s.descriptors, map: s.ranks)
        }
        // The grids `encodeTiledGroup` dispatches: 640 / 8 * 32 and 2560 / 8 * 32 threads in groups of 128, one row per record.
        let grid = grids.contents().advanced(by: offset).assumingMemoryBound(to: UInt32.self)
        let rows = UInt32(records.count)
        (grid[0], grid[1], grid[2], grid[3], grid[4], grid[5]) = (20, rows, 1, 80, rows, 1)
    }
    /// Check: a deferred group queued behind a wait does not run before its signal, then computes the ready group's
    /// bits from records it was given after encoding; an empty fill leaves its outputs untouched.
    package func deferredGroupParity(records: [Record], canonicalRanks: [Int], input: [Float], layer: Int = 7) throws {
        try validate(records, canonicalRanks, partial: true)
        func region() throws -> RegionBuffers {
            try RegionBuffers(
                gate: allocate(10 * 640 * 4), up: allocate(10 * 640 * 4), product: allocate(10 * 640 * 2),
                down: allocate(10 * 2560 * 4), descriptors: allocate(arguments.encodedLength), ranks: allocate(40))
        }
        let ready = try region()
        let deferred = try region()
        let x = try allocate(5120)
        let grids = try allocate(Self.deferredGridBytes)
        guard input.count == 2560, let event = device.makeSharedEvent() else { throw EngineError("Deferred group fixture unavailable") }
        let bf = x.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<2560 { bf[i] = BF16.encode(input[i]) }
        guard let immediate = queue.makeCommandBuffer() else { throw EngineError("Deferred group fixture unavailable") }
        try encodeReadyGroup(immediate, inputBF16: x, records: records, canonicalRanks: canonicalRanks, buffers: ready)
        immediate.commit()
        try immediate.waitBounded()
        let residency = try device.makeResidencySet(descriptor: MTLResidencySetDescriptor())
        var seen = Set<ObjectIdentifier>()
        residency.addAllocations(records.map(\.buffer).filter { seen.insert(ObjectIdentifier($0)).inserted })
        residency.commit()
        func run(_ fill: () throws -> Void) throws {
            guard let command = queue.makeCommandBuffer() else { throw EngineError("Deferred group fixture unavailable") }
            command.useResidencySet(residency)
            let value = event.signaledValue + 1
            command.encodeWaitForEvent(event, value: value)
            try encodeDeferredGroup(command, layer: layer, inputBF16: x, buffers: deferred, grids: grids, offset: 0)
            command.commit()
            usleep(2_000)
            guard command.status != .completed else { throw EngineError("Deferred expert group ran before its signal") }
            try fill()
            event.signaledValue = value
            try command.waitBounded()
            guard command.status == .completed else { throw EngineError.gpu("Deferred expert group failed", command.error) }
        }
        try run { try fillDeferredGroup(records: records, canonicalRanks: canonicalRanks, buffers: deferred, grids: grids, offset: 0) }
        for (a, b) in [(ready.product, deferred.product), (ready.down, deferred.down)] {
            guard memcmp(a.contents(), b.contents(), min(a.length, b.length)) == 0 else {
                throw EngineError("Deferred expert group differs from the ready group")
            }
        }
        let marker = deferred.down.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<(10 * 2560) { marker[i] = 1.5 }
        try run { try fillDeferredGroup(records: [], canonicalRanks: [], buffers: deferred, grids: grids, offset: 0) }
        guard (0..<(10 * 2560)).allSatisfy({ marker[$0] == 1.5 }) else { throw EngineError("An empty deferred expert group wrote outputs") }

        // GPU-filled group: the GPU fills the group from a residency table. Records 0..<n-2 were published before the command's
        // epoch (computed), record n-2 at it (skipped) and record n-1 is absent; the result is the ready group of the first.
        guard records.count >= 3 else { return }
        let keys = try allocate(48 * 512 * 8)
        let addresses = try allocate(64 * 8)
        let pairs = try allocate(80)
        memset(keys.contents(), 0, keys.length)
        memset(deferred.product.contents(), 0, deferred.product.length)
        memset(deferred.down.contents(), 0, deferred.down.length)
        let epoch: UInt32 = 3
        let trusted = records.count - 2
        let ids = (0..<10).map { (rank: Int) in (rank * 151 + 29) % 512 }
        let pair = pairs.contents().assumingMemoryBound(to: UInt32.self)
        for rank in 0..<10 {
            pair[rank * 2] = 0
            pair[rank * 2 + 1] = UInt32(ids[rank])
        }
        let entry = keys.contents().assumingMemoryBound(to: UInt64.self)
        let address = addresses.contents().assumingMemoryBound(to: UInt64.self)
        for (row, record) in records.enumerated() where row < records.count - 1 {
            address[row] = record.buffer.gpuAddress + UInt64(record.offset)
            entry[layer * 512 + ids[canonicalRanks[row]]] = UInt64(row + 1) | UInt64(row < trusted ? 1 : epoch) << 32
        }
        let expected = try region()
        memset(expected.product.contents(), 0, expected.product.length)
        memset(expected.down.contents(), 0, expected.down.length)
        guard let reference = queue.makeCommandBuffer() else { throw EngineError("Deferred group fixture unavailable") }
        try encodeReadyGroup(
            reference, inputBF16: x, records: Array(records.prefix(trusted)),
            canonicalRanks: Array(canonicalRanks.prefix(trusted)), buffers: expected)
        reference.commit()
        try reference.waitBounded()
        guard let lookup = queue.makeCommandBuffer() else { throw EngineError("Deferred group fixture unavailable") }
        lookup.useResidencySet(residency)
        try encodeResidentLookup(
            lookup, pairs: pairs, table: (keys, addresses), layer: layer, epoch: epoch, buffers: deferred, grids: grids, offset: 0)
        try encodeDeferredGroup(lookup, layer: layer, inputBF16: x, buffers: deferred, grids: grids, offset: 0)
        lookup.commit()
        try lookup.waitBounded()
        guard lookup.status == .completed else { throw EngineError.gpu("Resident group lookup failed", lookup.error) }
        guard grids.contents().load(fromByteOffset: 4, as: UInt32.self) == UInt32(trusted) else {
            throw EngineError("Resident group lookup trusted the wrong records")
        }
        for (a, b) in [(expected.product, deferred.product), (expected.down, deferred.down)] {
            guard memcmp(a.contents(), b.contents(), min(a.length, b.length)) == 0 else {
                throw EngineError("Resident group differs from the ready group")
            }
        }
    }
    private func floats(_ buffer: MTLBuffer, _ count: Int) -> [Float] {
        readbacks += UInt64(count * 4)
        return Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    }
    public func evaluate(
        _ input: [Float], records: [Record], canonicalRanks: [Int], weights: [Float],
        shouldContinue: () -> Bool = { true }, capture: ((String, [Float]) throws -> Void)? = nil
    ) throws -> [Float] {
        guard !active, input.count == 2560, weights.count == 10, weights.allSatisfy({ $0.isFinite && $0 >= 0 }),
            capture == nil || fused.gguf == nil
        else {
            throw EngineError("Invalid or reentrant gathered decode")
        }
        try validate(records, canonicalRanks)
        guard shouldContinue() else { throw IOOperationCancelled() }
        active = true
        let cold = scratch == nil
        defer { active = false }
        do {
            return try withExtendedLifetime(records) {
                try autoreleasepool {
                    let start = DispatchTime.now().uptimeNanoseconds
                    let s: Scratch
                    if let retained = scratch {
                        s = retained
                    } else {
                        s = try Scratch(
                            input: allocate(2560 * 2), gate: allocate(10 * 640 * 4), up: allocate(10 * 640 * 4),
                            product: allocate(10 * 640 * 2), down: allocate(10 * 2560 * 4), weights: allocate(10 * 4),
                            output: allocate(2560 * 4), descriptors: allocate(arguments.encodedLength), ranks: allocate(10 * 4))
                        scratch = s
                    }
                    bind(records, canonicalRanks, descriptors: s.descriptors, map: s.ranks)
                    let x = s.input.contents().assumingMemoryBound(to: UInt16.self)
                    for i in input.indices { x[i] = BF16.encode(input[i]) }
                    weights.withUnsafeBytes { s.weights.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
                    uploads += 5_200  // BF16 input5120 + FP32 scores40 + rank map40.
                    guard let command = queue.makeCommandBuffer() else { throw EngineError("Gathered command unavailable") }
                    command.label = "ten-expert-qmv-gate-up-product-down-canonical-reduce"
                    if let gguf = fused.gguf, let layout = ModelLayout.ggufExperts {
                        try gguf.encodeGroup(
                            command, input: s.input, records: records.map { ($0.buffer, $0.offset) }, ranks: canonicalRanks,
                            floats: s.product, gate: s.gate, up: s.up, down: s.down, layout: layout)
                    } else {
                        try encode(
                            command, input: s.input, descriptors: s.descriptors, ranks: s.ranks, records: records, kind: .gate,
                            output: s.gate)
                        try encode(
                            command, input: s.input, descriptors: s.descriptors, ranks: s.ranks, records: records, kind: .up, output: s.up)
                        try fused.encodeProduct(command, gate: s.gate, up: s.up, product: s.product, count: 10 * 640)
                        try encode(
                            command, input: s.product, descriptors: s.descriptors, ranks: s.ranks, records: records, kind: .down,
                            output: s.down)
                    }
                    try reductions.encodeWeightedExpertReduce10(command, input: s.down, weights: s.weights, output: s.output, tokens: 1)
                    prepareSeconds += Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                    try joined(command, shouldContinue: shouldContinue)
                    if let capture {
                        try capture("gate", floats(s.gate, 10 * 640))
                        try capture("up", floats(s.up, 10 * 640))
                        let bits = s.product.contents().assumingMemoryBound(to: UInt16.self)
                        try capture("product", (0..<(10 * 640)).map { BF16.decode(bits[$0]) })
                        try capture("down", floats(s.down, 10 * 2560))
                    }
                    return floats(s.output, 2560)
                }
            }
        } catch {
            // All submitted commands joined before leaving the autoreleasepool.
            // Newly admitted scratch rolls back on error; warm idle pages remain.
            if cold { scratch = nil }
            throw error
        }
    }

    /// Check helper: one projection (gate, up or down) for all ten records, with a guard band after the output.
    /// Down inputs are the canonical [10][640] products; gate and up share one 2560-value input.
    package func diagnosticProjection(_ input: [Float], records: [Record], canonicalRanks: [Int], kind: ExpertProjection) throws -> [Float]
    {
        guard !active, input.count == (kind == .down ? 10 * 640 : 2560) else { throw EngineError("Invalid gathered diagnostic input") }
        try validate(records, canonicalRanks)
        active = true
        defer { active = false }
        return try withExtendedLifetime(records) {
            try autoreleasepool {
                let x = try allocate(input.count * 2)
                let descriptors = try allocate(arguments.encodedLength)
                let ranks = try allocate(40)
                let size = 10 * kind.outputs * 4
                let output = try allocate(10 * kind.outputs * 4 + 128)
                memset(output.contents(), 0xa5, output.length)
                let bits = x.contents().assumingMemoryBound(to: UInt16.self)
                for i in input.indices { bits[i] = BF16.encode(input[i]) }
                bind(records, canonicalRanks, descriptors: descriptors, map: ranks)
                guard let command = queue.makeCommandBuffer() else { throw EngineError("Gathered diagnostic command unavailable") }
                try encode(command, input: x, descriptors: descriptors, ranks: ranks, records: records, kind: kind, output: output)
                try joined(command, shouldContinue: { true })
                let guardBytes = output.contents().assumingMemoryBound(to: UInt8.self)
                guard (size..<output.length).allSatisfy({ guardBytes[$0] == 0xa5 }) else {
                    throw EngineError("Gathered diagnostic output guard overwritten")
                }
                return floats(output, 10 * kind.outputs)
            }
        }
    }

    /// Standalone split-boundary check, with the same joined descriptor reuse as
    /// the whole executor. Original evaluate remains the independent schedule.
    package func diagnosticReadySplit(
        _ input: [Float], records: [Record], canonicalRanks: [Int],
        weights: [Float], firstIndices: [Int], independentBindings: Bool = false,
        groups explicitGroups: [[Int]]? = nil
    ) throws -> [Float] {
        try validate(records, canonicalRanks)
        let groups = (explicitGroups ?? [firstIndices, (0..<10).filter { !firstIndices.contains($0) }]).filter { !$0.isEmpty }
        guard !active, input.count == 2560, weights.count == 10,
            weights.allSatisfy({ $0.isFinite && $0 >= 0 }),
            Set(firstIndices).count == firstIndices.count,
            groups.flatMap({ $0 }).sorted() == Array(0..<10), !independentBindings || groups.count <= 2,
            firstIndices.allSatisfy({ (0..<10).contains($0) })
        else { throw EngineError("Invalid split fixture") }
        return try autoreleasepool {
            let x = try allocate(5120)
            let w = try allocate(40)
            let out = try allocate(10240 + 128)
            let s = try RegionBuffers(
                gate: allocate(25600), up: allocate(25600), product: allocate(12800),
                down: allocate(102400), descriptors: allocate(arguments.encodedLength), ranks: allocate(40))
            let alternate =
                independentBindings
                ? try RegionBuffers(
                    gate: s.gate, up: s.up, product: s.product, down: s.down,
                    descriptors: allocate(arguments.encodedLength), ranks: allocate(40)) : s
            var commands: [MTLCommandBuffer] = []
            defer { for command in commands { try? command.waitBounded() } }
            for b in s.all { memset(b.contents(), 0xa5, b.length) }
            memset(out.contents(), 0xa5, out.length)
            for i in input.indices { x.contents().assumingMemoryBound(to: UInt16.self)[i] = BF16.encode(input[i]) }
            weights.withUnsafeBytes { w.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            for (number, group) in groups.enumerated() {
                guard let command = queue.makeCommandBuffer() else { throw EngineError("Split fixture command unavailable") }
                try encodeReadyGroup(
                    command, inputBF16: x, records: group.map { records[$0] },
                    canonicalRanks: group.map { canonicalRanks[$0] }, buffers: number == 0 ? s : alternate)
                if number == groups.count - 1 {
                    try reductions.encodeWeightedExpertReduce10(command, input: s.down, weights: w, output: out, tokens: 1)
                }
                if independentBindings {
                    commands.append(command)
                    command.commit()
                } else {
                    try joined(command, shouldContinue: { true })
                }
            }
            for command in commands {
                try command.waitBounded()
                guard command.status == .completed else { throw EngineError.gpu("Independent group command failed", command.error) }
            }
            let bytes = out.contents().assumingMemoryBound(to: UInt8.self)
            guard (10240..<out.length).allSatisfy({ bytes[$0] == 0xa5 }) else { throw EngineError("Split output guard overwritten") }
            return floats(out, 2560)
        }
    }

    // qmv_pinned_reference's scalar/lane arithmetic is unchanged. Only input,
    // weight and output addressing adds the independent expert dimension.
    static let source = #"""
        // Queued-experts decode: one thread builds a layer's resident group from the router's ranks and the cache's GPU
        // residency table, then writes both dispatch grids (zero rows when nothing is resident). keys[layer * 512 + id]:
        // slot + 1 in the low 32 bits (0 = absent) and the epoch it was published in; slots[slot]: the record's address.
        kernel void expert_resident_group(device const uint *pairs [[buffer(0)]], device const ulong *keys [[buffer(1)]],
            device const ulong *slots [[buffer(2)]], constant uint2 &layerEpoch [[buffer(3)]], device ulong *records [[buffer(4)]],
            device uint *ranks [[buffer(5)]], device uint *grid [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
            if (tid != 0) return;
            uint n = 0;
            for (uint rank = 0; rank < 10; ++rank) {
                ulong entry = keys[layerEpoch.x * 512 + min(pairs[rank * 2 + 1], 511u)];
                uint slot = uint(entry & 0xffffffffu);
                if (slot != 0 && uint(entry >> 32) < layerEpoch.y) { records[n] = slots[slot - 1]; ranks[n] = rank; ++n; }
            }
            for (uint i = n; i < 10; ++i) { records[i] = 0; }
            grid[0] = 20; grid[1] = n; grid[2] = 1; grid[3] = 80; grid[4] = n; grid[5] = 1;
        }
        // Argument buffer of the ten expert records: raw GPU addresses, 80 bytes (the resident lookup writes them directly).
        struct ExpertRecords10 { array<device const uchar*, 10> values [[id(0)]]; };
        // Ten one-row QMVs in one grid: x = output row * 32 lanes, y = record. shape = [width, outputs, weight, scale and
        // bias byte offsets in the record, perLane (16 gate/up, 8 down), input stride (down reads its rank's product row)].
        kernel void expert_gathered_qmv10(device const ushort *sharedX [[buffer(0)]],
            constant ExpertRecords10 &records [[buffer(1)]], constant uint *ranks [[buffer(2)]],
            device float *out [[buffer(3)]], constant uint *shape [[buffer(4)]],
            uint2 tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
            uint rank = ranks[tid.y], row = tid.x / 32, width = shape[0], outputs = shape[1];
            device const ushort *x = sharedX + rank * shape[6];
            device const uchar *record = records.values[tid.y];
            device const uint *words = reinterpret_cast<device const uint *>(record + shape[2]);
            device const ushort *scales = reinterpret_cast<device const ushort *>(record + shape[3]);
            device const ushort *biases = reinterpret_cast<device const ushort *>(record + shape[4]);
            uint perLane = shape[5], lanes = 32, group = QGROUP;
            float result = 0.0f;
            if (lane < lanes) for (uint lo = lane*perLane; lo < width; lo += lanes*perLane) {
                float quantSum = 0.0f, biasSum = 0.0f;
                for (uint i = lo; i < min(lo+perLane, width); i += 4) {
                    float values[4], products[4];
                    for (uint j = 0; j < 4; ++j) {
                        values[j] = from_bf16(x[i+j]);
                        uint nibble = (words[row*(width/8)+(i+j)/8] >> (((i+j)%8)*4)) & 15;
                        products[j] = values[j] * QVAL(nibble);
                    }
                    biasSum = biasSum + round_bf16(round_bf16(round_bf16(values[0]+values[1])+values[2])+values[3]);
                    quantSum = quantSum + (((products[0]+products[1])+products[2])+products[3]);
                }
                uint index = row*(width/group)+lo/group;
                float dot = fma(QSCALE(scales, biases, index), quantSum, biasSum*QBIAS(biases, index));
                result = result + dot;
            }
            result = simd_sum(result);
            if (lane == 0) out[rank*outputs+row] = round_bf16(result);
        }
        """#
}
