import Darwin
import Foundation
import Metal
import ObjectiveC

/// An escrow carved from the existing scope-frontier reservation before any
/// state mutation. Metal permits retain the entire escrow through retirement;
/// escaping a buffer never permits either early uncharge or overlapping reuse.
public final class PrefillMoEBudget {
    private let ledger: MemoryLedger
    private let reservation: MemoryLedger.Reservation
    public let bytes: UInt64
    private let lock = NSCondition()
    private var used: UInt64 = 0
    private let metadata: UInt64 = PrefillMoEBudget.batchMetadataBytes
    /// How long a charge waits for live bytes to retire. A joined command's buffers retire when Metal drops its own
    /// reference to the command, which can trail `waitUntilCompleted` by a moment; the escrow has no slack beyond one
    /// batch's maps, so the next batch or layer could otherwise find the previous one's buffers still charged.
    package static let retireWait: TimeInterval = 2
    /// Charges that waited, and the longest wait (seconds): said once in the log when the escrow goes.
    private var waits = 0, longestWait = 0.0
    public static func pageRounded(_ bytes: Int) -> UInt64 {
        let page = UInt64(getpagesize())
        return (UInt64(bytes) + page - 1) / page * page
    }
    // Eight <=256-element UInt32 maps and <=8 small job/descriptor
    // collections (<=256 bytes/element), each under the strict <2x allocator
    // bound, plus 8 KiB object/closure bookkeeping. The reservation precedes
    // all jobs; reject an incompatible platform before any such allocation.
    public static var batchMetadataBytes: UInt64 { pageRounded(131072) }
    // A mixed scope adds at most one extra chunk per expert. Reserve 4 KiB
    // per extra encoded chunk, without allocating another activation buffer.
    package static let mixedTailLayerMetadataBytes: UInt64 = 512 * 4096
    public static func hostBytesForScope(tokens: Int) throws -> UInt64 {
        guard (1...8192).contains(tokens) else { throw EngineError("Invalid device MoE host map scope") }
        return pageRounded(tokens * 10 * 2) + pageRounded(tokens * 10) + pageRounded(2048 * 4)
    }
    public static func bytesForScope(tokens: Int) throws -> UInt64 {
        guard (1...8192).contains(tokens) else { throw EngineError("Invalid device MoE scope") }
        // Float scope input + canonical ten-rank Float output + eight per-job
        // UInt32 maps + bounded host coverage/ownership bookkeeping.
        return pageRounded(tokens * 2560 * 4) + pageRounded(tokens * 10 * 2560 * 4)
            + 8 * pageRounded(256 * 4) + (try hostBytesForScope(tokens: tokens)) + batchMetadataBytes
    }
    public init(ledger: MemoryLedger, bytes: UInt64) throws {
        let arrayBound =
            8 * 2 * malloc_good_size(256 * 4 + 256)
            + 8 * 2 * malloc_good_size(8 * 256 + 256) + 8192
        guard UInt64(arrayBound) <= Self.batchMetadataBytes, bytes > Self.batchMetadataBytes else {
            throw EngineError("Device MoE escrow is too small")
        }
        self.ledger = ledger
        self.bytes = bytes
        reservation = try ledger.reserve(bytes: bytes, category: "prefill_moe_scope_escrow")
    }
    deinit {
        try? ledger.release(reservation)
        if waits > 0 {
            FileHandle.standardError.write(
                Data(
                    String(
                        format: "origami: device MoE prefill: %d charges waited for a finished command's buffers (longest %.1f ms)\n",
                        waits, longestWait * 1000
                    ).utf8))
        }
    }
    fileprivate func take(bytes: UInt64) throws -> PrefillMoEPermit {
        lock.lock()
        defer { lock.unlock() }
        let started = Date()
        let deadline = started.addingTimeInterval(Self.retireWait)
        var waited = false
        while bytes > self.bytes - metadata - used {
            // Nothing live (or the wait ran out): the charge can never fit.
            guard used > 0, bytes <= self.bytes - metadata, Date() < deadline else {
                throw EngineError("Device MoE live buffers exceed owned scope escrow", code: "insufficient_memory")
            }
            waited = true
            _ = lock.wait(until: deadline)
        }
        // Only charges that waited and then fit: a refusal is not the race this counts.
        if waited {
            waits += 1
            longestWait = max(longestWait, Date().timeIntervalSince(started))
        }
        used += bytes
        return PrefillMoEPermit(budget: self, bytes: bytes)
    }
    fileprivate func retire(_ bytes: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        precondition(bytes <= used)
        used -= bytes
        lock.broadcast()
    }
    public var liveBytes: UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return used
    }
    /// A finished layer's contribution buffer, kept for the scope's next layer, so a whole-prefill scope allocates it
    /// once instead of once per layer (839 MB at 8,192 tokens). Its permit holds this budget: the scope drops it at its
    /// end (`dropSpare`).
    private var spare: MTLBuffer?
    fileprivate func contributions(bytes: Int, device: MTLDevice) throws -> MTLBuffer {
        do {
            lock.lock()
            let kept = spare
            spare = nil
            lock.unlock()
            if let kept, kept.length == bytes { return kept }
        }  // a kept buffer of another size is released here, before its share of the escrow is taken again
        return try PrefillMoEWorkspace.allocate(bytes: bytes, budget: self, device: device)
    }
    fileprivate func keep(_ buffer: MTLBuffer) {
        lock.lock()
        let previous = spare
        spare = buffer
        lock.unlock()
        withExtendedLifetime(previous) {}  // released outside the lock: its permit retires under it
    }
    package func dropSpare() {
        lock.lock()
        let kept = spare
        spare = nil
        lock.unlock()
        withExtendedLifetime(kept) {}
    }

    /// Whole-schedule frontiers share this same precharged escrow. The resource
    /// retains its permit, so adoption by a later MoE workspace cannot uncharge
    /// or double-charge a live allocation.
    package func buffer(bytes: Int, device: MTLDevice) throws -> MTLBuffer {
        try PrefillMoEWorkspace.allocate(bytes: bytes, budget: self, device: device)
    }
    fileprivate func owns(_ buffer: MTLBuffer) -> Bool {
        (objc_getAssociatedObject(buffer, &PrefillMoEAssociation.key) as? PrefillMoEPermit)?.budget === self
    }
}

/// A charge taken from a budget; attached to its buffer (associated object) so the bytes return when the last user retires.
private final class PrefillMoEPermit {
    let budget: PrefillMoEBudget, bytes: UInt64
    init(budget: PrefillMoEBudget, bytes: UInt64) {
        self.budget = budget
        self.bytes = bytes
    }
    deinit { budget.retire(bytes) }
}
private enum PrefillMoEAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// Exact page ownership avoids Swift Array's capacity derived from oversized
/// cached malloc blocks. The complete region is charged before anonymous mmap.
private final class PrefillMoEHostMap {
    let count: Int
    let expected: UnsafeMutablePointer<UInt16>
    let covered: UnsafeMutablePointer<UInt8>
    let pending: UnsafeMutablePointer<UInt32>
    private let memory: UnsafeMutableRawPointer
    private let bytes: Int
    private let permit: PrefillMoEPermit
    init(tokens: Int, budget: PrefillMoEBudget) throws {
        let count = tokens * 10
        let bytes = Int(try PrefillMoEBudget.hostBytesForScope(tokens: tokens))
        let permit = try budget.take(bytes: UInt64(bytes))
        guard let memory = mmap(nil, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0), memory != MAP_FAILED else {
            throw EngineError("Device MoE host map allocation failed", code: "insufficient_memory")
        }
        self.memory = memory
        self.bytes = bytes
        self.permit = permit
        self.count = count
        expected = memory.bindMemory(to: UInt16.self, capacity: count)
        covered = memory.advanced(by: Int(PrefillMoEBudget.pageRounded(count * 2))).bindMemory(to: UInt8.self, capacity: count)
        pending = memory.advanced(by: Int(PrefillMoEBudget.pageRounded(count * 2) + PrefillMoEBudget.pageRounded(count)))
            .bindMemory(to: UInt32.self, capacity: 2048)
        expected.initialize(repeating: UInt16.max, count: count)
        covered.initialize(repeating: 0, count: count)
        pending.initialize(repeating: 0, count: 2048)
    }
    deinit { precondition(munmap(memory, bytes) == 0, "Owned device MoE host map failed to retire") }
}

/// Integer-address gather/scatter between a scope's token rows and its canonical [token][rank] slots, plus the
/// counters of the device MoE prefill. Existing expert/reduction shaders and compilation modes do not change.
/// Compile once per executor, not per layer.
public final class PrefillMoETransfer {
    /// Cumulative device MoE counters (layers initialized or adopted, batches, rows, bytes, failures, reductions).
    public struct Snapshot: Codable, Sendable {
        public let initializedLayers, completedBatches, completedExpertRows, uploadedInputBytes: UInt64
        public let failedBatches, completedReductions, scatteredOutputBytes: UInt64
    }
    private var initializedLayers: UInt64 = 0, completedBatches: UInt64 = 0
    private var completedExpertRows: UInt64 = 0, uploadedInputBytes: UInt64 = 0
    private var failedBatches: UInt64 = 0, completedReductions: UInt64 = 0
    public func snapshot() -> Snapshot {
        Snapshot(
            initializedLayers: initializedLayers, completedBatches: completedBatches,
            completedExpertRows: completedExpertRows, uploadedInputBytes: uploadedInputBytes,
            failedBatches: failedBatches, completedReductions: completedReductions,
            scatteredOutputBytes: completedExpertRows * 2560 * 4)
    }
    fileprivate func didInitialize(tokens: Int) {
        initializedLayers += 1
        uploadedInputBytes += UInt64(tokens * 2560 * 4)
    }
    fileprivate func didAdopt() { initializedLayers += 1 }
    fileprivate func didComplete(rows: Int) {
        completedBatches += 1
        completedExpertRows += UInt64(rows)
    }
    fileprivate func didFail() { failedBatches += 1 }
    fileprivate func didReduce() { completedReductions += 1 }
    package let device: MTLDevice
    private let gatherPipeline, scatterPipeline: MTLComputePipelineState
    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Device MoE Metal unavailable") }
        self.device = device
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        let library = try device.makeLibrary(source: Self.source, options: options)
        guard let gather = library.makeFunction(name: "moe_indexed_gather"),
            let scatter = library.makeFunction(name: "moe_indexed_scatter")
        else {
            throw EngineError("Device MoE transfer kernels unavailable")
        }
        gatherPipeline = try device.makeComputePipelineState(function: gather)
        scatterPipeline = try device.makeComputePipelineState(function: scatter)
    }
    func encode(
        _ command: MTLCommandBuffer, gather: Bool, source: MTLBuffer, map: MTLBuffer,
        destination: MTLBuffer, rows: Int, mapOffset: Int = 0,
        encoder supplied: MTLComputeCommandEncoder? = nil
    ) throws {
        guard command.device.registryID == device.registryID,
            source.device.registryID == device.registryID, map.device.registryID == device.registryID,
            destination.device.registryID == device.registryID, (1...256).contains(rows),
            mapOffset >= 0, mapOffset % 4 == 0, mapOffset <= map.length, map.length - mapOffset >= rows * 4,
            let encoder = supplied ?? command.makeComputeCommandEncoder()
        else {
            throw EngineError("Device MoE transfer geometry/encoder failure")
        }
        let pipeline = gather ? gatherPipeline : scatterPipeline
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(map, offset: mapOffset, index: 1)
        encoder.setBuffer(destination, offset: 0, index: 2)
        var count = UInt32(rows * 2560)
        withUnsafeBytes(of: &count) { encoder.setBytes($0.baseAddress!, length: $0.count, index: 3) }
        encoder.dispatchThreads(
            MTLSize(width: rows * 2560, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        if supplied == nil { encoder.endEncoding() }
    }
    static let source = #"""
        #include <metal_stdlib>
        using namespace metal;
        // moe_indexed_gather: one thread per element (count = rows * 2560). Output row r is the FP32 scope input row of
        // token slots[r] / 10 (slot = token * 10 + rank), narrowed to BF16 with round-to-nearest-even in integer
        // arithmetic; a NaN keeps sign and payload and gains the quiet bit. Bit-identical to BF16.encode (BITWISE).
        kernel void moe_indexed_gather(device const uint *input [[buffer(0)]], device const uint *slots [[buffer(1)]],
            device ushort *output [[buffer(2)]], constant uint &count [[buffer(3)]], uint index [[thread_position_in_grid]]) {
            if (index >= count) return;
            uint bits = input[(slots[index / 2560] / 10) * 2560 + index % 2560];
            // BF16.encode: preserve NaN sign/payload and force the quiet bit;
            // otherwise round-to-nearest-even using integer arithmetic only.
            output[index] = (bits & 0x7fffffffU) > 0x7f800000U ? ushort((bits >> 16) | 0x40U)
                : ushort((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
        }
        // moe_indexed_scatter: copies FP32 expert output row r to row slots[r] of the canonical [token][rank][2560]
        // contribution buffer. A pure copy (BITWISE); the host map lets each slot be written once per layer.
        kernel void moe_indexed_scatter(device const uint *input [[buffer(0)]], device const uint *slots [[buffer(1)]],
            device uint *output [[buffer(2)]], constant uint &count [[buffer(3)]], uint index [[thread_position_in_grid]]) {
            if (index < count) output[slots[index / 2560] * 2560 + index % 2560] = input[index];
        }
        """#
}

/// One layer's device MoE state for a prefill scope of T tokens: the FP32 input [T][2560] (uploaded, or adopted from
/// the whole-prefill scope), the canonical FP32 contribution buffer [T][10][2560] indexed by slot = token * 10 + rank,
/// and a host map holding, per slot, the expert that must write it and whether it is written (0 empty, 2 staged by
/// the current batch or layer, 1 complete). Writes are accepted only for slots their expert owns, once each, and the
/// weighted ten-rank reduction (PinnedReductions' fixed order) runs only when every slot is complete, so the result
/// does not depend on expert or batch order (BITWISE). The T * 102,400-byte contribution buffer is allocated per layer,
/// except on the whole-prefill scope, whose layers pass one on (`PrefillMoEBudget.contributions`): a layer hands it on
/// only once its reduction read every slot and finished, and the next layer writes every slot again before reducing.
/// The final layer of a scope whose unread rows skip their routed experts (`stale`) is the one exception: slots of
/// experts outside its union keep the previous layer's values (covered 3), which only unread rows' reductions see.
public final class PrefillMoEWorkspace {
    public let tokens: Int
    package let input, contributions: MTLBuffer
    private let budget: PrefillMoEBudget
    private let transfer: PrefillMoETransfer
    private let host: PrefillMoEHostMap
    private var pendingCount = 0
    private var active = false, invalidated = false, activeLayer = false
    private var encodedReduction: MTLCommandBuffer?
    private var encodedReductionFinished = false
    /// The reduction finished (no command still reads the contributions) and the buffer is the scope's to pass on.
    private var reduced = false, passesOn = false
    public private(set) var completedSlots = 0
    /// Slots that keep the passed-on buffer's previous values (`stale`); counted complete from the start.
    let staleSlots: Int

    public init(
        input: [Float], selectedRows: [Int: [(Int, Int)]],
        budget: PrefillMoEBudget, transfer: PrefillMoETransfer
    ) throws {
        guard !input.isEmpty, input.count % 2560 == 0, input.count / 2560 <= 8192 else {
            throw EngineError("Invalid device MoE input")
        }
        let tokens = input.count / 2560
        let host = try PrefillMoEHostMap(tokens: tokens, budget: budget)
        for (expert, positions) in selectedRows {
            guard (0..<512).contains(expert), !positions.isEmpty else { throw EngineError("Invalid device MoE expert map") }
            var previous = -1
            for (token, rank) in positions {
                guard (0..<tokens).contains(token), (0..<10).contains(rank), token > previous,
                    host.expected[token * 10 + rank] == UInt16.max
                else {
                    throw EngineError("Device MoE destinations overlap, reorder or exceed scope")
                }
                previous = token
                host.expected[token * 10 + rank] = UInt16(expert)
            }
        }
        guard (0..<host.count).allSatisfy({ host.expected[$0] != UInt16.max }) else { throw EngineError("Device MoE router map has holes") }
        self.tokens = tokens
        self.budget = budget
        self.transfer = transfer
        self.host = host
        staleSlots = 0
        self.input = try Self.allocate(bytes: input.count * 4, budget: budget, device: transfer.device)
        contributions = try Self.allocate(bytes: tokens * 10 * 2560 * 4, budget: budget, device: transfer.device)
        input.withUnsafeBytes { self.input.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        transfer.didInitialize(tokens: tokens)
    }
    /// Adopt a whole-scope input already produced on device, under this exact
    /// escrow. Contributions/maps/coverage use the existing implementation.
    /// `stale`: slots no expert of `selectedRows` writes keep the passed-on buffer's values instead of being holes.
    package init(
        tokens: Int, input: MTLBuffer, selectedRows: [Int: [(Int, Int)]], budget: PrefillMoEBudget, transfer: PrefillMoETransfer,
        stale: Bool = false
    ) throws {
        guard (1...8192).contains(tokens), input.length >= tokens * 2560 * 4,
            input.device.registryID == transfer.device.registryID, budget.owns(input)
        else {
            throw EngineError("Device MoE input is not owned by its scope escrow")
        }
        let host = try PrefillMoEHostMap(tokens: tokens, budget: budget)
        for (expert, positions) in selectedRows {
            guard (0..<512).contains(expert), !positions.isEmpty else { throw EngineError("Invalid device MoE expert map") }
            var previous = -1
            for (token, rank) in positions {
                guard (0..<tokens).contains(token), (0..<10).contains(rank), token > previous,
                    host.expected[token * 10 + rank] == UInt16.max
                else { throw EngineError("Device MoE destinations overlap, reorder or exceed scope") }
                previous = token
                host.expected[token * 10 + rank] = UInt16(expert)
            }
        }
        var staleSlots = 0
        for i in 0..<host.count where host.expected[i] == UInt16.max {
            guard stale else { throw EngineError("Device MoE router map has holes") }
            host.covered[i] = 3
            staleSlots += 1
        }
        self.tokens = tokens
        self.budget = budget
        self.transfer = transfer
        self.host = host
        self.input = input
        self.staleSlots = staleSlots
        completedSlots = staleSlots
        contributions = try budget.contributions(bytes: tokens * 10 * 2560 * 4, device: transfer.device)
        passesOn = true
        transfer.didAdopt()  // No activation upload occurred.
    }
    fileprivate static func allocate(bytes: Int, budget: PrefillMoEBudget, device: MTLDevice) throws -> MTLBuffer {
        guard bytes > 0, bytes <= device.maxBufferLength else { throw EngineError("Device MoE allocation extent invalid") }
        let charged = PrefillMoEBudget.pageRounded(bytes)
        let permit = try budget.take(bytes: charged)
        guard let result = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw EngineError("Device MoE buffer allocation failed", code: "insufficient_memory")
        }
        objc_setAssociatedObject(result, &PrefillMoEAssociation.key, permit, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        guard UInt64(result.allocatedSize) <= charged else {
            throw EngineError("Device MoE buffer exceeded its page reservation", code: "insufficient_memory")
        }
        return result
    }
    package func beginBatch(experts: [Int], slots: [[UInt32]]) throws {
        guard !active, !invalidated, (1...8).contains(slots.count), experts.count == slots.count else {
            throw EngineError("Invalid or overlapping device MoE batch")
        }
        // Caller job arrays are allocated under the already-admitted 128 KiB
        // metadata bound. Verify their actual capacity before using them.
        let mapArrayBound = 2 * malloc_good_size(256 * 4 + 256)
        guard experts.capacity <= 128, slots.capacity <= 128,
            slots.allSatisfy({ $0.capacity <= mapArrayBound / 4 })
        else {
            throw EngineError("Device MoE batch metadata exceeds its admitted Array capacity")
        }
        var staged = 0
        do {
            for (index, positions) in slots.enumerated() {
                guard (1...256).contains(positions.count) else { throw EngineError("Invalid device MoE tile") }
                var previous = -1
                for slot in positions {
                    guard Int(slot) < host.count, Int(host.expected[Int(slot)]) == experts[index],
                        host.covered[Int(slot)] == 0, Int(slot) / 10 > previous, staged < 2048
                    else {
                        throw EngineError("Device MoE batch repeats or changes a canonical destination")
                    }
                    previous = Int(slot) / 10
                    host.pending[staged] = slot
                    staged += 1
                    host.covered[Int(slot)] = 2
                }
            }
        } catch {
            for index in 0..<staged { host.covered[Int(host.pending[index])] = 0 }
            throw error
        }
        pendingCount = staged
        active = true
    }
    /// Build the complete immutable destination map before any dispatch. Full
    /// coverage uses the owned bitmap itself, not the small-batch pending array.
    /// A rejected pre-commit map rolls back every staged bit.
    func withLayerMetadata<T>(bytes: UInt64, _ body: () throws -> T) throws -> T {
        let permit = try budget.take(bytes: bytes)
        return try withExtendedLifetime(permit, body)
    }
    func beginLayer(
        experts: [Int], selectedRows: [Int: [(Int, Int)]], map: MTLBuffer,
        ungroupedTailStart: Int? = nil, partial: Bool = false
    ) throws -> Int {
        guard !active, !invalidated, partial || completedSlots == staleSlots, (1...512).contains(experts.count),
            experts.count == selectedRows.count, experts.capacity <= 1024,
            map.device.registryID == transfer.device.registryID, map.length == host.count * 4
        else {
            throw EngineError("Invalid or overlapping device MoE layer")
        }
        if let tail = ungroupedTailStart {
            guard tail > 0, tail % 256 == 0, tail < tokens, tokens - tail < 256 else {
                throw EngineError("Invalid mixed expert arithmetic boundary")
            }
        }
        var staged = 0
        var tiles = 0
        var previousExpert = -1
        let slots = map.contents().assumingMemoryBound(to: UInt32.self)
        do {
            for expert in experts {
                guard (0..<512).contains(expert), expert > previousExpert,
                    let positions = selectedRows[expert], (1...tokens).contains(positions.count)
                else {
                    throw EngineError("Layer map expert union is invalid")
                }
                previousExpert = expert
                if let tail = ungroupedTailStart {
                    let split = positions.firstIndex { $0.0 >= tail } ?? positions.count
                    tiles += (split + 255) / 256 + (positions.count - split + 255) / 256
                } else {
                    tiles += (positions.count + 255) / 256
                }
                var previousToken = -1
                for (token, rank) in positions {
                    guard (0..<tokens).contains(token), (0..<10).contains(rank), token > previousToken else {
                        throw EngineError("Layer map rows are invalid or reordered")
                    }
                    let slot = token * 10 + rank
                    guard staged < host.count, host.covered[slot] == 0, Int(host.expected[slot]) == expert else {
                        throw EngineError("Layer map repeats or changes a canonical destination")
                    }
                    slots[staged] = UInt32(slot)
                    staged += 1
                    host.covered[slot] = 2
                    previousToken = token
                }
            }
            guard partial ? staged > 0 && staged <= host.count - completedSlots : staged + staleSlots == host.count,
                tiles <= (ungroupedTailStart == nil ? 830 : 1342)
            else {
                throw EngineError("Layer map has holes or exceeds its dispatch bound")
            }
        } catch {
            for i in 0..<host.count where host.covered[i] == 2 { host.covered[i] = 0 }
            throw error
        }
        pendingCount = staged
        active = true
        activeLayer = true
        return tiles
    }
    func completeLayer(success: Bool) {
        precondition(active && activeLayer)
        if success {
            for i in 0..<host.count where host.covered[i] == 2 { host.covered[i] = 1 }
            completedSlots += pendingCount
            transfer.didComplete(rows: pendingCount)
        } else {
            invalidated = true
            transfer.didFail()
        }
        pendingCount = 0
        active = false
        activeLayer = false
    }

    package func completeBatch(success: Bool) {
        precondition(active && !activeLayer)
        if success {
            for index in 0..<pendingCount { host.covered[Int(host.pending[index])] = 1 }
            completedSlots += pendingCount
            transfer.didComplete(rows: pendingCount)
        } else {
            invalidated = true
            transfer.didFail()
        }
        pendingCount = 0
        active = false
    }
    /// Stale slots of tokens `token...`: a mixed tail's QMV rows cover the rest.
    func staleSlots(from token: Int) -> Int { staleSlots == 0 ? 0 : (token * 10..<host.count).filter { host.covered[$0] == 3 }.count }
    package func mapBuffer(_ slots: [UInt32]) throws -> MTLBuffer {
        guard active, !invalidated else { throw EngineError("Device MoE map outside active batch") }
        let result = try Self.allocate(bytes: slots.count * 4, budget: budget, device: transfer.device)
        slots.withUnsafeBytes { result.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return result
    }
    package func encodeGather(
        _ command: MTLCommandBuffer, map: MTLBuffer, output: MTLBuffer, rows: Int,
        mapOffset: Int = 0, encoder: MTLComputeCommandEncoder? = nil
    ) throws {
        guard active, !invalidated, output.length >= rows * 2560 * 2 else { throw EngineError("Invalid device MoE gather") }
        try transfer.encode(
            command, gather: true, source: input, map: map, destination: output, rows: rows, mapOffset: mapOffset, encoder: encoder)
    }
    package func encodeScatter(
        _ command: MTLCommandBuffer, map: MTLBuffer, input: MTLBuffer, rows: Int,
        mapOffset: Int = 0, encoder: MTLComputeCommandEncoder? = nil
    ) throws {
        guard active, !invalidated, input.length >= rows * 2560 * 4 else { throw EngineError("Invalid device MoE scatter") }
        try transfer.encode(
            command, gather: false, source: input, map: map, destination: contributions, rows: rows, mapOffset: mapOffset, encoder: encoder)
    }
    public func reduce(weights: [Float], using reductions: PinnedReductions) throws -> [Float] {
        guard !active, !invalidated, completedSlots == tokens * 10, (0..<host.count).allSatisfy({ host.covered[$0] & 1 == 1 }) else {
            throw EngineError("Device MoE contributions are incomplete or invalidated")
        }
        let result = try reductions.weightedExpertReduce10(buffer: contributions, tokens: tokens, weights: weights)
        transfer.didReduce()
        reduced = true
        return result
    }

    package func encodeReduction(_ command: MTLCommandBuffer, weights: MTLBuffer, output: MTLBuffer, using reductions: PinnedReductions)
        throws
    {
        guard !active, !invalidated, encodedReduction == nil, !encodedReductionFinished,
            completedSlots == tokens * 10, (0..<host.count).allSatisfy({ host.covered[$0] & 1 == 1 }),
            command.status == .notEnqueued, command.commandQueue.device.registryID == transfer.device.registryID,
            weights.device.registryID == transfer.device.registryID, output.device.registryID == transfer.device.registryID,
            output !== contributions, output !== weights, output !== input
        else {
            throw EngineError("Device MoE encoded reduction has incomplete contributions or invalid ownership")
        }
        try reductions.encodeWeightedExpertReduce10(command, input: contributions, weights: weights, output: output, tokens: tokens)
        encodedReduction = command
    }
    package func completeEncodedReduction(_ command: MTLCommandBuffer) throws {
        guard encodedReduction === command, command.status == .completed, !encodedReductionFinished else {
            throw EngineError("Device MoE reduction publication is unjoined or owned by another command")
        }
        encodedReduction = nil
        encodedReductionFinished = true
        transfer.didReduce()
        reduced = true
    }
    deinit { if reduced && passesOn { budget.keep(contributions) } }
}
